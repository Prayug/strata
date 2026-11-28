"""Second, separate implementation (duckdb SQL straight on the parquet files) of every figure that decides a verdict.
Nothing here reuses common.py."""
import duckdb

D = "/Users/prayugsigdel/Coding/strata/data/market/datasets"
con = duckdb.connect()
con.execute("SET TimeZone='UTC'")

for sym in ("SPY", "QQQ", "IWM"):
    con.execute(f"""
    CREATE OR REPLACE TABLE d_{sym} AS
    SELECT date, open, high, low, close,
           lag(close) OVER (ORDER BY date) AS pc,
           lag(close, 2) OVER (ORDER BY date) AS pc2,
           lag(high) OVER (ORDER BY date) AS ph,
           lag(low) OVER (ORDER BY date) AS pl,
           row_number() OVER (ORDER BY date) AS i
    FROM read_parquet('{D}/yahoo/{sym}/underlying/trade/1d/v1/data/daily.parquet')
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE m_{sym} AS
    SELECT CAST(timezone('America/New_York', ts) AS DATE) AS date,
           CAST(extract('hour' FROM timezone('America/New_York', ts)) * 60 + extract('minute' FROM timezone('America/New_York', ts)) AS INTEGER) AS mod,
           open, high, low, close
    FROM read_parquet('{D}/alpaca/{sym}/underlying/trade/1m/v1/data/*.parquet')
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE s_{sym} AS
    SELECT date, count(*) AS n, min(mod) AS first, max(mod) AS last,
           arg_min(open, mod) AS m_open, max(high) AS m_high, min(low) AS m_low, arg_max(close, mod) AS m_close,
           (min(mod) = 570 AND max(mod) IN (959, 779)) AS whole
    FROM m_{sym} WHERE mod >= 570 AND mod < 960 GROUP BY date
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE b_{sym} AS
    SELECT d.*, coalesce(s.whole, false) AS whole, s.m_open, s.m_high, s.m_low, s.m_close,
           CASE WHEN coalesce(s.whole, false) THEN s.m_open ELSE d.open END AS so
    FROM d_{sym} d LEFT JOIN s_{sym} s USING (date)
    """)


def q(sql):
    return con.execute(sql).fetchall()


print("whole sessions SPY/QQQ/IWM:", q("SELECT (SELECT count(*) FROM s_SPY WHERE whole), (SELECT count(*) FROM s_QQQ WHERE whole), (SELECT count(*) FROM s_IWM WHERE whole)"))

print("\n-- day counts behind the '0 of N' items")
print("IWM days with prev close since 2000-01-01:", q("SELECT count(*) FROM b_IWM WHERE pc IS NOT NULL AND date >= DATE '2000-01-01'"))
print("SPY days with prev close:", q("SELECT count(*) FROM b_SPY WHERE pc IS NOT NULL"))
print("SPY days since 2016-01-01:", q("SELECT count(*) FROM b_SPY WHERE pc IS NOT NULL AND date >= DATE '2016-01-01'"))
print("SPY opened below prev close (session open):", q("SELECT count(*) FROM b_SPY WHERE so < pc"))
print("QQQ opened below prev close (session open):", q("SELECT count(*) FROM b_QQQ WHERE so < pc"))
print("SPY whole sessions with a 14:29 bar:", q("SELECT count(*) FROM s_SPY s JOIN m_SPY m USING (date) WHERE s.whole AND m.mod = 869"))
for name, sql in {
    "IWM open->close since 2000": "SELECT count(*), avg(close/so-1)*100, sum((close>so)::INT), sum((close<so)::INT) FROM b_IWM WHERE pc IS NOT NULL",
    "SPY open->close since 2016": "SELECT count(*), avg(close/so-1)*100, sum((close>so)::INT), sum((close<so)::INT) FROM b_SPY WHERE pc IS NOT NULL AND date >= DATE '2016-01-01'",
    "SPY open->close all": "SELECT count(*), avg(close/so-1)*100, sum((close>so)::INT), sum((close<so)::INT) FROM b_SPY WHERE pc IS NOT NULL",
    "SPY gap<0 open->close": "SELECT count(*), avg(close/so-1)*100, sum((close>so)::INT), sum((close<so)::INT) FROM b_SPY WHERE so < pc",
    "QQQ gap<0 open->close": "SELECT count(*), avg(close/so-1)*100, sum((close>so)::INT), sum((close<so)::INT) FROM b_QQQ WHERE so < pc",
    "SPY 14:30->close": "SELECT count(*), avg(b.close/m.close-1)*100, sum((b.close>m.close)::INT), sum((b.close<m.close)::INT) FROM b_SPY b JOIN m_SPY m USING (date) WHERE b.whole AND m.mod = 869",
}.items():
    print(f"  {name}: n, mean %, ended higher, ended lower =", q(sql))

print("\n-- sam/sh-02#1 (12:00)")
print("any whole session, close > open:", q("SELECT count(*), sum((close > so)::INT) FROM b_SPY WHERE whole"))
print("below the open at 12:00 (11:59 bar close < 09:30 open), and of those close > open:",
      q("SELECT count(*), sum((b.close > b.so)::INT) FROM b_SPY b JOIN m_SPY m USING (date) WHERE b.whole AND m.mod = 719 AND m.close < b.so"))

print("\n-- alex/ah-27#1 (previous low)")
print("daily low <  prev daily low:", q("SELECT count(*), sum((low < pl)::INT), sum((low <= pl)::INT) FROM b_SPY WHERE pc IS NOT NULL"))
print("minute low where whole else daily, vs prev daily low (<, <=):",
      q("SELECT sum(((CASE WHEN whole THEN m_low ELSE low END) < pl)::INT), sum(((CASE WHEN whole THEN m_low ELSE low END) <= pl)::INT) FROM b_SPY WHERE pc IS NOT NULL"))
print("minute low vs previous session's minute low where stored (<, <=):", q("""
    WITH x AS (SELECT *, CASE WHEN whole THEN m_low ELSE low END AS cl,
                      lag(CASE WHEN whole THEN m_low ELSE low END) OVER (ORDER BY date) AS pml FROM b_SPY)
    SELECT sum((cl < pml)::INT), sum((cl <= pml)::INT) FROM x WHERE pc IS NOT NULL"""))

print("\n-- morgan/mh-01#1 and sam/sh-21#1 (daily record)")
print("SPY high>prev high; and close<prev high:", q("SELECT count(*), sum((close < ph)::INT) FROM b_SPY WHERE pc IS NOT NULL AND high > ph"))
print("SPY 2016+: low<prev low; and close>prev close:", q("SELECT count(*), sum((close > pc)::INT) FROM b_SPY WHERE pc IS NOT NULL AND date >= DATE '2016-01-01' AND low < pl"))

print("\n-- movement items")
con.execute(f"""CREATE OR REPLACE TABLE vix AS SELECT date, close AS v FROM read_parquet('{D}/cboe/VIX/underlying/*/1d/v1/data/daily.parquet')""")
for K, extra, name in [(5, "", "morgan/mh-18#1"), (10, "AND pv > 35", "jordan/jh-16#1")]:
    con.execute(f"""
    CREATE OR REPLACE TABLE w AS
    WITH x AS (
      SELECT b.*, lag(vix.v) OVER (ORDER BY b.date) AS pv,
             lead(b.close, {K}) OVER (ORDER BY b.date) AS c_k,
             lead(b.close, {K - 1}) OVER (ORDER BY b.date) AS c_k1
      FROM b_SPY b LEFT JOIN vix USING (date))
    SELECT * FROM x WHERE pc2 IS NOT NULL AND pc / pc2 - 1 <= -0.02 {extra}
    """)
    print(name, "matching sessions:", q("SELECT count(*), count(c_k) FROM w"))
    for lab, expr in [("open[t] -> close[t+K]  (Strata's figures)", "c_k / so - 1"),
                      ("close[t] -> close[t+K]", "c_k / close - 1"),
                      ("close[t-1] -> close[t+K-1]", "c_k1 / pc - 1"),
                      ("close[t-1] -> close[t+K]", "c_k / pc - 1")]:
        print(f"   {lab}: n, mean %, median %, higher %, q25, q75 =",
              q(f"""SELECT count(*), round(avg({expr})*100, 3), round(median({expr})*100, 3), round(avg(({expr} > 0)::INT)*100, 2),
                           round(quantile_cont({expr}, 0.25)*100, 3), round(quantile_cont({expr}, 0.75)*100, 3)
                    FROM w WHERE c_k IS NOT NULL"""))

print("\n-- alex/ah-24#1 (IWM down streak)")
con.execute("""
CREATE OR REPLACE TABLE iw AS
WITH x AS (SELECT *, lag(close,1) OVER (ORDER BY date) c1, lag(close,2) OVER (ORDER BY date) c2, lag(close,3) OVER (ORDER BY date) c3,
                  lag(close,4) OVER (ORDER BY date) c4, lag(close,5) OVER (ORDER BY date) c5,
                  lead(close,5) OVER (ORDER BY date) f5, lead(close,4) OVER (ORDER BY date) f4 FROM b_IWM)
SELECT *, (c1 < c2 AND c2 < c3 AND c3 < c4 AND c4 < c5) AS streak4 FROM x
""")
print("sessions after 4+ down closes:", q("SELECT count(*) FROM iw WHERE streak4"))
for lab, ev in [("close 5 later > previous close", "f5 > pc"), ("close 5 later > session open of t", "f5 > so"),
                ("that session closes above its own open", "close > so"), ("close 4 later > previous close", "f4 > pc")]:
    need = "f5 IS NOT NULL" if "f5" in ev else ("f4 IS NOT NULL" if "f4" in ev else "TRUE")
    print(f"   {lab}: streak k/n =", q(f"SELECT sum(({ev})::INT), count(*) FROM iw WHERE streak4 AND {need}"),
          " any day k/n =", q(f"SELECT sum(({ev})::INT), count(*) FROM iw WHERE pc IS NOT NULL AND {need}"))

print("\n-- casey/ch-07#1 (IWM gap fill since 2016)")
print("gap>=1% (session open vs prev daily close), fills by minute lows:",
      q("SELECT count(*), sum(((CASE WHEN whole THEN m_low ELSE low END) <= pc)::INT) FROM b_IWM WHERE date >= DATE '2016-01-01' AND so / pc - 1 >= 0.01"))

print("\n-- touch items")
print("QQQ +1.5% from session open:", q("SELECT count(*), sum(((CASE WHEN whole THEN m_high ELSE high END) >= so * 1.015)::INT) FROM b_QQQ WHERE pc IS NOT NULL"))
print("SPY +5% from session open:", q("SELECT count(*), sum(((CASE WHEN whole THEN m_high ELSE high END) >= so * 1.05)::INT) FROM b_SPY WHERE pc IS NOT NULL"))
print("QQQ +1% from session open (plain):", q("SELECT count(*), sum(((CASE WHEN whole THEN m_high ELSE high END) >= so * 1.01)::INT) FROM b_QQQ WHERE pc IS NOT NULL"))
print("QQQ +1% before -0.5%:", q("""
WITH pre AS (SELECT sum((high >= open*1.01 AND low > open*0.995)::INT) AS hit, sum((high >= open*1.01 AND low <= open*0.995)::INT) AS both, count(*) AS n
             FROM b_QQQ WHERE pc IS NOT NULL AND NOT whole),
     t AS (SELECT b.date, min(CASE WHEN m.high >= b.so*1.01 THEN m.mod END) AS tu, min(CASE WHEN m.low <= b.so*0.995 THEN m.mod END) AS td
           FROM b_QQQ b JOIN m_QQQ m USING (date) WHERE b.whole AND b.pc IS NOT NULL GROUP BY b.date),
     post AS (SELECT sum((tu IS NOT NULL AND (td IS NULL OR tu < td))::INT) AS hit, sum((tu IS NOT NULL AND td IS NOT NULL AND tu = td)::INT) AS both, count(*) AS n FROM t)
SELECT pre.hit + post.hit AS happened, pre.both + post.both AS undecidable, pre.n + post.n - pre.hit - post.hit - pre.both - post.both AS didnt FROM pre, post"""))

print("\n-- streak items / close above previous close")
for sym, thr in (("IWM", 5), ("SPY", 3), ("SPY", 5)):
    conds = " AND ".join([f"lag(close,{k}) OVER w > lag(close,{k + 1}) OVER w" for k in range(1, thr + 1)])
    print(f"{sym} after {thr}+ up closes: n, closes above prev close =",
          q(f"WITH x AS (SELECT close, pc, ({conds}) AS s FROM b_{sym} WINDOW w AS (ORDER BY date)) SELECT count(*), sum((close > pc)::INT) FROM x WHERE s"))
print("QQQ close>prev close:", q("SELECT count(*), sum((close > pc)::INT) FROM b_QQQ WHERE pc IS NOT NULL"))
print("IWM / SPY any day close>prev close:", q("SELECT sum((close > pc)::INT), count(*) FROM b_IWM WHERE pc IS NOT NULL"), q("SELECT sum((close > pc)::INT), count(*) FROM b_SPY WHERE pc IS NOT NULL"))
