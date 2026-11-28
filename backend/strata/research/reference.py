"""Independent reference calculations: a second, separate route to the numbers Strata reports.

Written apart from the engines (strata.engine, strata.research.study / profile / agreement and the backtester). It reads the
stored Parquet files directly with DuckDB SQL, and every definition it uses is written out below, so a number the
application reports can be recomputed without any of the code that produced it, and a difference between the two can be
traced to the days and the cause.

Definitions (choices, stated with every result):
  session         a regular-hours trading day in the daily record; intraday bars are the regular-hours minute bars
                  (09:30-16:00 New York), each stamped with its START time
  open            "minute" (default, as the application): the open of the session's first minute bar where minute bars
                  are stored, otherwise the daily record's open. "official": always the daily record's open
  close           the daily record's close (the official closing print)
  price at T      the close of the last minute bar that starts before T (the price when the clock reads T); at the open,
                  the open; at the close, the official close
  window [S, E)   the minute bars that start at or after S and before E; a high or low inside it "touches" a level
  minute bars     Alpaca's 1-minute trade bars, used for a session only when they cover it (first bar within 15 minutes of
                  the open, last bar ending within 15 minutes of the close); otherwise the session has its daily record only
  previous_return the previous session's close against the close before it; prev_otc its close against its own open
  gap             the open against the previous close;  change at T: price at T against the previous close
  from_open at T  price at T against the open;  first N-minute candle: price at 09:30+N against the open
  streaks         consecutive sessions closing above (up) or below (down) the close before, ending the previous session
  VIX             the previous session's VIX close (the daily VIX record)
  directions      up means > 0 and down < 0; a session where a compared move is exactly zero counts as neither

Nothing here is used to produce an answer. It is used to check answers (`check_frame`, the V9 evaluation) and to explain a
difference when there is one.
"""
from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path

import duckdb

ET = "America/New_York"
MIN = lambda hhmm: int(hhmm[:2]) * 60 + int(hhmm[3:5])  # noqa: E731


def _root() -> Path:
    from strata.config import get_settings
    return get_settings().data_dir / "market" / "datasets"


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


class Reference:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or _root())
        self.con = duckdb.connect()
        self.loaded: set[str] = set()

    # ------------------------------------------------------------------------------------------------------- loading
    def _daily_file(self, sym: str) -> Path | None:
        order = ("cboe", "yahoo") if sym == "VIX" else ("yahoo", "alpaca", "github_eod", "cboe", "synthetic")  # the VIX record is Cboe's own
        for prov in order:
            for p in sorted(self.root.glob(f"{prov}/{sym}/underlying/*/1d/v*/data/daily.parquet")):
                return p
        for p in sorted(self.root.glob(f"*/{sym}/underlying/*/1d/v*/data/daily.parquet")):
            return p  # another installation's (or a test store's) only daily record
        return None

    def _minute_files(self, sym: str) -> list[Path]:
        """Alpaca's 1-minute trade bars only: the stored, verified history. Other providers' recent minute bars (Yahoo keeps
        about a month) are left out, so a session is never assembled from two feeds."""
        found = sorted(self.root.glob(f"alpaca/{sym}/underlying/trade/1m/v*/data/[12][0-9][0-9][0-9].parquet"))
        if found:
            return found
        # no Alpaca bars (a test store, another installation): the one provider that holds one-minute bars, never two mixed
        for prov in sorted({p.parts[len(self.root.parts)] for p in self.root.glob(f"*/{sym}/underlying/trade/1m/v*/data/*.parquet")}):
            return sorted(self.root.glob(f"{prov}/{sym}/underlying/trade/1m/v*/data/[12][0-9][0-9][0-9].parquet"))
        return []

    def q(self, sql: str, params: list | None = None) -> list[tuple]:
        return self.con.execute(sql, params or []).fetchall()

    def load(self, sym: str) -> None:
        sym = sym.upper()
        if sym in self.loaded:
            return
        dp = self._daily_file(sym)
        if dp is None:
            raise LookupError(f"no daily record stored for {sym}")
        self.con.execute(f"create or replace table d_{sym} as select cast(date as date) date, open::double d_open, high::double d_high, low::double d_low, close::double d_close "
                         f"from read_parquet('{dp}') where close is not null order by date")
        mf = self._minute_files(sym)
        if mf:
            files = ", ".join(f"'{p}'" for p in mf)
            self.con.execute(f"""create or replace table b_{sym} as
                select cast(timezone('{ET}', ts) as date) date,
                       (extract(hour from timezone('{ET}', ts)) * 60 + extract(minute from timezone('{ET}', ts)))::int m,
                       open::double o, high::double h, low::double l, close::double c, coalesce(volume, 0)::double v
                from read_parquet([{files}]) order by ts""")
            # a session's minute bars are used only when they cover it: the first bar within 15 minutes of the open and the last
            # ending within 15 minutes of the close (13:00 on an early-close day). A fragment would give a wrong high, low and
            # close; such a session is known from its daily record alone. (The same rule as the application's bar store.)
            self.con.execute(f"""create or replace table cover_{sym} as select date from b_{sym} group by date
                                 having min(m) <= 585 and (max(m) >= 944 or (max(m) between 764 and 779))""")
            self.con.execute(f"delete from b_{sym} where date not in (select date from cover_{sym})")
            self.con.execute(f"create or replace table mo_{sym} as select date, arg_min(o, m) m_open, arg_max(c, m) m_close, max(h) m_high, min(l) m_low from b_{sym} group by date")
        else:
            self.con.execute(f"create or replace table b_{sym} (date date, m int, o double, h double, l double, c double, v double)")
            self.con.execute(f"create or replace table mo_{sym} (date date, m_open double, m_close double, m_high double, m_low double)")
        self.con.execute(f"""create or replace table s_{sym} as
            with x as (
              select d.date, d.d_open, d.d_high, d.d_low, d.d_close close, m.m_open, m.m_close, (m.m_open is not null) has_minute,
                     coalesce(m.m_open, d.d_open) open,
                     lag(d.d_close) over w prev_close, lag(d.d_close, 2) over w prev2_close, lag(d.d_open) over w prev_d_open,
                     lag(coalesce(m.m_open, d.d_open)) over w prev_open, lag(greatest(d.d_high, coalesce(m.m_high, d.d_high))) over w prev_high,
                     lag(least(d.d_low, coalesce(m.m_low, d.d_low))) over w prev_low, isodow(d.date) dow
              from d_{sym} d left join mo_{sym} m using (date) window w as (order by d.date))
            select *, (close / prev_close - 1) * 100 ret, (prev_close / prev2_close - 1) * 100 prev_ret, (prev_close / prev_open - 1) * 100 prev_otc,
                   (open / prev_close - 1) * 100 gap, (d_open / prev_close - 1) * 100 d_gap, (close / open - 1) * 100 otc, (close / d_open - 1) * 100 d_otc,
                   row_number() over (order by date) - 1 i
            from x order by date""")
        # streaks ending the previous session: runs of closes above / below the close before
        rows = self.q(f"select date, ret from s_{sym} order by date")
        up = dn = 0
        out = []
        for dte, r in rows:
            out.append((dte, up, dn))  # as known at this session's open: the run up to the previous session
            if r is None or (isinstance(r, float) and math.isnan(r)):
                up = dn = 0
            elif r > 0:
                up, dn = up + 1, 0
            elif r < 0:
                up, dn = 0, dn + 1
            else:
                up = dn = 0
        self.con.execute(f"create or replace table k_{sym} (date date, up_streak int, down_streak int)")
        self.con.executemany(f"insert into k_{sym} values (?, ?, ?)", out)
        if sym != "VIX" and self._daily_file("VIX") is not None:
            self.load("VIX")
            self.con.execute(f"create or replace table v_{sym} as select s.date, lag(v.close) over (order by s.date) vix_prev from s_{sym} s left join s_VIX v using (date)")
        else:
            self.con.execute(f"create or replace table v_{sym} as select date, null::double vix_prev from s_{sym}")
        self.loaded.add(sym)

    # ------------------------------------------------------------------------------------------------- intraday bits
    def _price_at(self, sym: str, hhmm: str, alias: str) -> str:
        """SQL for a per-date table (date, <alias>): the price when the clock reads hhmm."""
        if hhmm in ("open", "09:30"):
            return f"select date, open {alias} from s_{sym}"
        if hhmm == "close":
            return f"select date, close {alias} from s_{sym}"
        m = MIN(hhmm)
        return f"select date, arg_max(c, m) {alias} from b_{sym} where m < {m} group by date"

    def _window(self, sym: str, s: str, e: str) -> str:
        """SQL for (date, w_high, w_low, w_bars): the minute bars in [s, e). A window that is the whole session is also known
        on a day with only a daily record: the day's high and low (that is how a gap fill before 2016 is decided)."""
        ms = 570 if s in ("open", "09:30") else MIN(s)
        me = 960 if e == "close" else MIN(e)
        minute = f"select date, max(h) w_high, min(l) w_low, count(*) w_bars from b_{sym} where m >= {ms} and m < {me} group by date"
        if ms == 570 and me == 960:
            return (f"select date, greatest(d_high, close) w_high, least(d_low, close) w_low, 1 w_bars from s_{sym} where not has_minute "
                    f"union all select w.date, greatest(w.w_high, s.close) w_high, least(w.w_low, s.close) w_low, w.w_bars from ({minute}) w join s_{sym} s using (date)")
        return minute

    # ---------------------------------------------------------------------------------------------------- conditions
    def _cond_sql(self, c: dict, sym: str, at: str, joins: dict) -> str | None:
        """A V8 frame condition as a SQL predicate on the sessions table `s` (with joined per-date tables)."""
        k, op, v = c.get("key"), c.get("op") or ">", c.get("value")
        sqlop = {"<=": "<=", ">=": ">=", "<": "<", ">": ">", "=": "="}.get(op, op)
        val = 0.0 if v is None or isinstance(v, str) else float(v)
        if k == "prev_return":
            if op == "abs>=":
                return f"abs(s.prev_ret) >= {val}"
            return f"s.prev_ret {sqlop} {val}"
        if k == "prev_otc":
            return f"s.prev_otc {sqlop} {val}"
        if k == "gap":
            return f"s.gap {sqlop} {val}"
        if k == "weekday":
            return f"s.dow = {['Mon', 'Tue', 'Wed', 'Thu', 'Fri'].index(v) + 1}"
        if k == "vix":
            joins["vx"] = f"v_{sym}"
            return f"vx.vix_prev {sqlop} {val}"
        if k in ("up_streak", "down_streak"):
            joins["ks"] = f"k_{sym}"
            return f"ks.{k} {sqlop} {val}"
        if k == "first_candle":
            mins = int(c.get("minutes") or 5)
            alias = f"fc{mins}"
            joins[alias] = f"({self._price_at(sym, _hhmm(570 + mins), 'p')})"
            d = c.get("direction")
            if d == "up":
                return f"{alias}.p > s.open"
            if d == "down":
                return f"{alias}.p < s.open"
            return f"{alias}.p <> s.open"
        if k in ("from_open", "change"):
            alias = "pa_" + re.sub(r"\W", "", at)
            joins[alias] = f"({self._price_at(sym, at, 'p')})"
            base = "s.open" if k == "from_open" else "s.prev_close"
            return f"({alias}.p / {base} - 1) * 100 {sqlop} {val}"
        return None

    # -------------------------------------------------------------------------------------------------------- frames
    def frequency(self, sym: str, conditions: list[dict], outcome: dict, start: str = "open", end: str = "close", period=(None, None),
                  open_source: str = "minute") -> dict:
        """How often the outcome happened on the sessions that met the conditions. Returns n, k, pct and the dates."""
        sym = sym.upper()
        self.load(sym)
        joins: dict = {}
        at = start if start not in ("signal",) else "open"
        where = []
        for c in conditions:
            w = self._cond_sql(c, sym, at, joins)
            if w is None:
                raise ValueError(f"the reference does not implement condition {c.get('key')!r}")
            where.append(w)
        o = outcome or {}
        okey = o.get("key")
        sessions_end = None
        if isinstance(end, str) and end.startswith("sessions:"):
            sessions_end = int(end.split(":")[1])
        ps = f"({self._price_at(sym, start if start not in ('signal',) else 'open', 'p')})"
        joins["ps"] = ps
        open_col = "s.open" if open_source == "minute" else "s.d_open"
        if okey == "close_vs" or okey == "event_close":
            ref = {"open": open_col, "prev_close": "s.prev_close", "start": "ps.p"}.get(o.get("ref") or "open", open_col)
            if sessions_end:
                endp = f"(select close from s_{sym} z where z.i = s.i + {sessions_end - 1})"
                ref = {"prev_close": "s.prev_close", "open": open_col, "start": "ps.p"}.get(o.get("ref") or "prev_close", "s.prev_close")
            elif end == "close":
                endp = "s.close"
            else:
                joins["pe"] = f"({self._price_at(sym, end, 'p')})"
                endp = "pe.p"
            d = o.get("direction") or "up"
            hit = f"{endp} > {ref}" if d == "up" else f"{endp} < {ref}"
            valid = f"{endp} is not null and {ref} is not null"
        elif okey == "touch":
            joins["w"] = f"({self._window(sym, start, end)})"
            ref = {"open": open_col, "prev_close": "s.prev_close", "start": "ps.p", "entry": "ps.p"}.get(o.get("ref") or "start", "ps.p")
            if (o.get("ref") or "start") == "start" and start in ("open", "09:30"):
                ref = open_col
            lvl = f"{ref} * (1 + {float(o['value'])} / 100)"
            hit = f"w.w_high >= {lvl}" if float(o["value"]) > 0 else f"w.w_low <= {lvl}"
            valid = "w.w_bars > 0"
        elif okey == "touch_level":
            joins["w"] = f"({self._window(sym, start, end)})"
            level = {"prev_close": "s.prev_close", "prev_high": "s.prev_high", "prev_low": "s.prev_low", "open": open_col}[o.get("level") or "prev_close"]
            d = o.get("direction")
            if d is None:  # a gap fill: back to the level from whichever side the session opened on
                hit = f"((s.open > {level} and w.w_low <= {level}) or (s.open < {level} and w.w_high >= {level}) or s.open = {level})"
            elif d == "up":
                hit = f"w.w_high >= {level}"
            else:
                hit = f"w.w_low <= {level}"
            valid = "w.w_bars > 0"
        elif okey == "event":
            ev = o.get("event") or {}
            w = self._cond_sql(ev, sym, at, joins)
            hit, valid = w, "true"
        else:
            raise ValueError(f"the reference does not implement outcome {okey!r}")
        if period and period[0]:
            where.append(f"s.date >= DATE '{str(period[0])[:10]}'")
        if period and len(period) > 1 and period[1]:
            where.append(f"s.date <= DATE '{str(period[1])[:10]}'")
        if any(c.get("key") in ("prev_return", "prev_otc") for c in conditions):
            where.append("s.prev2_close is not null")
        else:
            where.append("s.prev_close is not null")
        js = " ".join(f"left join {t} {a} using (date)" for a, t in joins.items())
        sql = f"select s.date, ({hit}) hit from s_{sym} s {js} where {' and '.join(where) or 'true'} and ({valid})"
        rows = [(d, bool(h)) for d, h in self.q(sql) if h is not None]
        n, k = len(rows), sum(1 for _, h in rows if h)
        return {"n": n, "k": k, "pct": (100.0 * k / n) if n else None, "dates": [str(d) for d, _ in rows], "hits": [str(d) for d, h in rows if h]}

    def agreement(self, sym: str, minutes: int, conditions: list[dict] | None = None, period=(None, None)) -> dict:
        """The first N-minute candle's direction against the rest of the session and against the whole session."""
        sym = sym.upper()
        self.load(sym)
        joins: dict = {"fc": f"({self._price_at(sym, _hhmm(570 + minutes), 'p')})"}
        where = ["s.has_minute", "fc.p is not null"]
        for c in conditions or []:
            w = self._cond_sql(c, sym, _hhmm(570 + minutes), joins)
            if w is None:
                raise ValueError(f"the reference does not implement condition {c.get('key')!r}")
            where.append(w)
        if period and period[0]:
            where.append(f"s.date >= DATE '{str(period[0])[:10]}'")
        if period and len(period) > 1 and period[1]:
            where.append(f"s.date <= DATE '{str(period[1])[:10]}'")
        js = " ".join(f"left join {t} {a} using (date)" for a, t in joins.items())
        rows = self.q(f"select s.date, sign(fc.p - s.open) c, sign(s.close - fc.p) r, sign(s.close - s.open) d from s_{sym} s {js} where {' and '.join(where)}")
        # a session whose candle did not move predicts nothing and is left out; a rest (or day) that ends exactly flat
        # counts as a miss, not as agreement (the application counts it the same way)
        rest = [(c, r) for _, c, r, _ in rows if c != 0]
        day = [(c, d) for _, c, _, d in rows if c != 0]
        agree = lambda xs: sum(1 for a, b in xs if a == b)  # noqa: E731
        return {"n": len(rows), "rest_n": len(rest), "rest_agree": agree(rest), "rest_pct": 100.0 * agree(rest) / len(rest) if rest else None,
                "day_n": len(day), "day_agree": agree(day), "day_pct": 100.0 * agree(day) / len(day) if day else None}

    def movement(self, sym: str, conditions: list[dict], start: str = "open", end: str = "close", period=(None, None)) -> dict:
        """The average and median move (percent) from the start to the end, on the sessions that met the conditions."""
        sym = sym.upper()
        self.load(sym)
        joins: dict = {"ps": f"({self._price_at(sym, start, 'p')})"}
        endp = "s.close"
        if end != "close":
            joins["pe"] = f"({self._price_at(sym, end, 'p')})"
            endp = "pe.p"
        where = ["ps.p is not null", f"{endp} is not null", "s.prev2_close is not null"]
        for c in conditions:
            w = self._cond_sql(c, sym, start, joins)
            if w is None:
                raise ValueError(f"the reference does not implement condition {c.get('key')!r}")
            where.append(w)
        if period and period[0]:
            where.append(f"s.date >= DATE '{str(period[0])[:10]}'")
        js = " ".join(f"left join {t} {a} using (date)" for a, t in joins.items())
        r = self.q(f"select count(*), avg(({endp} / ps.p - 1) * 100), median(({endp} / ps.p - 1) * 100) from s_{sym} s {js} where {' and '.join(where)}")[0]
        return {"n": int(r[0]), "mean": r[1], "median": r[2]}

    def extreme_bucket(self, sym: str, which: str = "high", bucket: int = 30, conditions: list[dict] | None = None) -> dict:
        """In which part of the session the day's high (or low) was made most often (first time it was reached)."""
        sym = sym.upper()
        self.load(sym)
        joins: dict = {}
        where = ["s.has_minute"]
        for c in conditions or []:
            w = self._cond_sql(c, sym, "open", joins)
            where.append(w)
        js = " ".join(f"left join {t} {a} using (date)" for a, t in joins.items())
        agg = "max(h)" if which == "high" else "min(l)"
        col = "h" if which == "high" else "l"
        rows = self.q(f"""with e as (select b.date, {agg} x from b_{sym} b group by b.date),
                              t as (select b.date, min(b.m) m from b_{sym} b join e using (date) where b.{col} = e.x group by b.date)
                          select ((t.m - 570) // {bucket}) k, count(*) from t join s_{sym} s using (date) {js} where {' and '.join(where)} group by 1 order by 1""")
        tot = sum(n for _, n in rows)
        best = max(rows, key=lambda x: x[1]) if rows else None
        return {"sessions": tot, "bucket": _hhmm(570 + int(best[0]) * bucket) if best else None, "share": 100.0 * best[1] / tot if tot else None,
                "shares": {_hhmm(570 + int(k) * bucket): 100.0 * n / tot for k, n in rows}}

    def weekday_means(self, sym: str, measure: str = "open_to_close", period=(None, None)) -> dict:
        sym = sym.upper()
        self.load(sym)
        col = {"open_to_close": "s.otc", "return": "s.ret", "close_to_close": "s.ret"}.get(measure, "s.otc")
        where = ["s.prev_close is not null"]
        if period and period[0]:
            where.append(f"s.date >= DATE '{str(period[0])[:10]}'")
        rows = self.q(f"select s.dow, count(*), avg({col}) from s_{sym} s where {' and '.join(where)} group by 1 order by 1")
        names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday"}
        out = {names[d]: {"n": n, "mean": m} for d, n, m in rows if d in names}
        best = max(out.items(), key=lambda kv: kv[1]["mean"]) if out else (None, {})
        return {"by": out, "weekday": best[0], "mean": best[1].get("mean")}

    # ------------------------------------------------------------------------------- V10: the profile answers, recalculated
    def _where(self, sym: str, conditions: list[dict] | None, period, joins: dict) -> list[str]:
        where = []
        if period and period[0]:
            where.append(f"s.date >= DATE '{str(period[0])[:10]}'")
        if period and len(period) > 1 and period[1]:
            where.append(f"s.date <= DATE '{str(period[1])[:10]}'")
        for c in conditions or []:
            if c.get("key") == "rel":
                continue
            w = self._cond_sql(c, sym, "open", joins)
            if w is None:
                raise ValueError(f"condition {c.get('key')!r} is not covered by the reference")
            where.append(w)
        return where

    def relation(self, sym: str, outcome: dict, conditions: list[dict] | None = None, period=(None, None), then: dict | None = None) -> dict:
        """How often one price of the session stood above or below a level (profile.relation_frequency), counted in SQL.
        A price at a time is the close of the last minute bar that ended by then; the previous session's high and low are
        the daily record's."""
        sym = sym.upper()
        self.load(sym)
        conds = list(conditions or [])
        timed: dict = {}
        cj: dict = {}

        def feat(f: str) -> str:
            if ":" not in f:
                return {"open": "s.open", "close": "s.close", "high": "s.d_high", "low": "s.d_low", "prev_close": "s.prev_close", "prev_high": "p.ph", "prev_low": "p.pl"}[f]
            kind, t = f.split(":", 1)
            mm = MIN(t)
            alias = "f_" + re.sub(r"\W", "", f)
            timed[alias] = {"at": (f"select date, arg_max(c, m) v from b_{sym} where m < {mm} group by date" if mm > 570 else f"select date, arg_min(o, m) v from b_{sym} group by date"),
                            "high_by": f"select date, max(h) v from b_{sym} where m < {mm} group by date", "low_by": f"select date, min(l) v from b_{sym} where m < {mm} group by date",
                            "high_from": f"select date, max(h) v from b_{sym} where m >= {mm} group by date", "low_from": f"select date, min(l) v from b_{sym} where m >= {mm} group by date"}[kind]
            return f"{alias}.v"

        def pred(r: dict) -> str:
            a, b = feat(r["a"]), feat(r["b"])
            if r.get("pct"):
                b = f"({b} * {1 + float(r['pct']) / 100})"
            return f"{a} {r['op']} {b}"

        where = ["s.prev_close is not null", "p.ph is not null", "p.pl is not null"] + self._where(sym, conds, period, cj)
        where += [pred(c["rel"]) for c in conds if c.get("key") == "rel"]
        hit = pred(outcome)
        hit2 = pred(then) if then else "false"
        js = " ".join(f"join ({q}) {a} using (date)" for a, q in timed.items()) + " " + " ".join(f"left join {t} {a} using (date)" for a, t in cj.items())
        rows = self.q(f"""select s.date, ({hit}) hit, ({hit2}) hit2 from s_{sym} s
                          join (select date, lag(d_high) over (order by date) ph, lag(d_low) over (order by date) pl from s_{sym}) p using (date) {js}
                          where {' and '.join(where)} order by s.date""")
        hits = [str(d) for d, h, _ in rows if h]
        return {"n": len(rows), "k": len(hits), "dates": [str(d) for d, _, _ in rows], "hits": hits, "then_k": sum(1 for _, h, h2 in rows if h and h2) if then else None}

    def bucket_stats(self, sym: str, measure: str = "abs_move", bucket: int = 30, conditions: list[dict] | None = None, period=(None, None)) -> dict:
        """Per part of the session: where the day's high or low was first made (share of sessions), or the average size of
        the move / of the high-to-low range in that part (profile.timing)."""
        sym = sym.upper()
        self.load(sym)
        joins: dict = {}
        where = ["s.has_minute"] + self._where(sym, conditions, period, joins)
        js = " ".join(f"left join {t} {a} using (date)" for a, t in joins.items())
        if measure in ("high_time", "low_time"):
            agg, col = ("max(h)", "h") if measure == "high_time" else ("min(l)", "l")
            rows = self.q(f"""with e as (select b.date, {agg} x from b_{sym} b group by b.date),
                                  t as (select b.date, min(b.m) m from b_{sym} b join e using (date) where b.{col} = e.x group by b.date)
                              select ((t.m - 570) // {bucket}) k, count(*), null from t join s_{sym} s using (date) {js} where {' and '.join(where)} group by 1 order by 1""")
            tot = sum(n for _, n, _ in rows)
            return {"sessions": tot, "by": {_hhmm(570 + int(k) * bucket): {"n": n, "share": 100.0 * n / tot} for k, n, _ in rows}}
        expr = "(x.hi - x.lo) / x.o * 100" if measure == "range" else "abs(x.c / x.o - 1) * 100"
        rows = self.q(f"""with x as (select b.date, ((b.m - 570) // {bucket}) k, arg_min(b.o, b.m) o, arg_max(b.c, b.m) c, max(b.h) hi, min(b.l) lo from b_{sym} b group by 1, 2)
                          select x.k, count(*), avg({expr}) from x join s_{sym} s using (date) {js} where {' and '.join(where)} group by 1 order by 1""")
        n_days = self.q(f"select count(*) from s_{sym} s {js} where {' and '.join(where)}")[0][0]
        return {"sessions": int(n_days), "by": {_hhmm(570 + int(k) * bucket): {"n": n, "mean": float(m)} for k, n, m in rows}}

    def group_stats(self, sym: str, by: str = "weekday", conditions: list[dict] | None = None, period=(None, None)) -> dict:
        """Per weekday or month: sessions, the average open-to-close and close-to-close move, and the share of sessions that
        closed above / below the open and the previous close (profile.breakdown)."""
        sym = sym.upper()
        self.load(sym)
        joins: dict = {}
        where = ["s.prev_close is not null"] + self._where(sym, conditions, period, joins)
        js = " ".join(f"left join {t} {a} using (date)" for a, t in joins.items())
        key = "s.dow" if by == "weekday" else "month(s.date)"
        rows = self.q(f"""select {key} g, count(*), avg(s.otc), avg(s.ret), avg((s.otc > 0)::int) * 100, avg((s.otc < 0)::int) * 100, avg((s.ret > 0)::int) * 100, avg((s.ret < 0)::int) * 100
                          from s_{sym} s {js} where {' and '.join(where)} group by 1 order by 1""")
        names = ({1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday"} if by == "weekday" else
                 {i + 1: n for i, n in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])})
        return {"by": {names[g]: {"n": n, "oc_mean": a, "cc_mean": b, "oc_up": c, "oc_down": d_, "cc_up": e, "cc_down": f} for g, n, a, b, c, d_, e, f in rows if g in names}}

    def event(self, sym: str, event: dict, conditions: list[dict] | None = None, period=(None, None)) -> dict:
        """How often a gap or a day's move of a given size happened (profile.event_frequency)."""
        sym = sym.upper()
        self.load(sym)
        joins: dict = {}
        col = {"gap": "s.gap", "day_return": "s.ret", "prev_return": "s.ret"}.get(event.get("key"))
        if col is None:
            raise ValueError("event not covered")
        op = {"<=": "<=", ">=": ">=", "<": "<", ">": ">"}[event.get("op")]
        where = ["s.prev_close is not null"] + self._where(sym, conditions, period, joins)
        js = " ".join(f"left join {t} {a} using (date)" for a, t in joins.items())
        n, k = self.q(f"select count(*), count(*) filter (where {col} {op} {float(event.get('value') or 0)}) from s_{sym} s {js} where {' and '.join(where)}")[0]
        return {"n": int(n), "k": int(k)}

    # ------------------------------------------------------------------------------------------ check one answer
    def check_frame(self, frame: dict) -> dict | None:
        """Recompute a V8 question frame's headline number independently. None when this reference does not cover the
        frame (the answer then says it was not independently checked)."""
        kind = frame.get("kind")
        sym = frame.get("symbol") or "SPY"
        period = tuple(frame.get("period") or (None, None))
        conds = [c for c in frame.get("conditions") or [] if not c.get("values")]
        if len(conds) != len(frame.get("conditions") or []):
            return None  # several sizes measured at once: each variant is its own run
        try:
            if kind == "frequency" and frame.get("outcome"):
                o = dict(frame["outcome"])
                if o.get("key") in ("recover", "reverse", "continue"):
                    return None
                return {"kind": "frequency", **self.frequency(sym, conds, o, frame.get("start") or "open", frame.get("end") or "close", period)}
            if kind == "agreement" and frame.get("predictor") == "first_candle":
                return {"kind": "agreement", "by_candle": {m: self.agreement(sym, int(m), conds, period) for m in frame.get("candle") or [5]}}
            if kind == "movement":
                return {"kind": "movement", **self.movement(sym, conds, frame.get("start") or "open", frame.get("end") or "close", period)}
        except (ValueError, KeyError, LookupError, duckdb.Error):
            return None
        return None


_REF: Reference | None = None


def reference() -> Reference:
    global _REF
    if _REF is None:
        _REF = Reference()
    return _REF


# ----------------------------------------------------------------------------------- the V9 evaluation's answers
def answer_value(name: str, args: dict, ref: Reference | None = None) -> dict:
    """The independently calculated values for one item of evalset_v9.SEALED_ANSWERS (and the development answers)."""
    R = ref or reference()
    s = args.get("symbol", "SPY")
    if name == "close_above_open":
        r = R.frequency(s, [], {"key": "close_vs", "ref": "open", "direction": "up"}, period=(args.get("start"), args.get("end")))
        return {"pct": r["pct"], "k": r["k"], "n": r["n"]}
    if name == "gap_fill":
        g = float(args["gap"])
        r = R.frequency(s, [{"key": "gap", "op": "<=" if g < 0 else ">=", "value": g}], {"key": "touch_level", "level": "prev_close", "direction": None})
        return {"pct": r["pct"], "k": r["k"], "n": r["n"]}
    if name == "after_prev_return":
        p = float(args["prev"])
        oc = {"close<open": ("open", "down"), "close>open": ("open", "up"), "close>prev_close": ("prev_close", "up"), "close<prev_close": ("prev_close", "down")}[args["outcome"]]
        r = R.frequency(s, [{"key": "prev_return", "op": "<=" if p < 0 else ">=", "value": p}], {"key": "close_vs", "ref": oc[0], "direction": oc[1]})
        return {"pct": r["pct"], "k": r["k"], "n": r["n"]}
    if name == "candle_agreement":
        r = R.agreement(s, int(args["minutes"]))
        return {"rest_pct": r["rest_pct"], "day_pct": r["day_pct"], "n": r["n"]}
    if name == "best_weekday":
        r = R.weekday_means(s)
        return {"weekday": r["weekday"], "mean": r["mean"]}
    if name == "window_mean_after_gap":
        g = float(args["gap"])
        r = R.movement(s, [{"key": "gap", "op": ">=" if g > 0 else "<=", "value": g}], args.get("start", "open"), args.get("end", "close"))
        return {"mean": r["mean"], "n": r["n"]}
    if name == "gap_count":
        g, y = float(args["gap"]), int(args["year"])
        r = R.frequency(s, [], {"key": "event", "event": {"key": "gap", "op": "<" if g < 0 else ">", "value": g}}, period=(f"{y}-01-01", f"{y}-12-31"))
        return {"k": r["k"], "n": r["n"], "pct": r["pct"]}
    if name == "extreme_bucket":
        r = R.extreme_bucket(s, args.get("which", "low"))
        return {"bucket": r["bucket"], "share": r["share"]}
    if name == "touch_before":
        r = R.frequency(s, [], {"key": "touch", "value": float(args["pct"]), "ref": "open"}, "open", args["until"])
        return {"pct": r["pct"], "k": r["k"], "n": r["n"]}
    if name == "after_streak":
        key = "up_streak" if args.get("direction") == "up" else "down_streak"
        oc = {"close<prev_close": ("prev_close", "down"), "close>prev_close": ("prev_close", "up"), "close<open": ("open", "down"), "close>open": ("open", "up")}[args["outcome"]]
        r = R.frequency(s, [{"key": key, "op": ">=", "value": int(args["n"])}], {"key": "close_vs", "ref": oc[0], "direction": oc[1]})
        return {"pct": r["pct"], "k": r["k"], "n": r["n"]}
    if name == "option_backtest":
        from strata.research.reference_options import option_backtest
        return option_backtest(args, R)
    raise KeyError(name)
