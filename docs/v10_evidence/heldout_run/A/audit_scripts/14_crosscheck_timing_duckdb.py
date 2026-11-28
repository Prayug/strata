"""Second implementation (duckdb SQL) of the half-hour items and the weekday item.  Does not reuse common.py."""
import duckdb

D = "/Users/prayugsigdel/Coding/strata/data/market/datasets"
con = duckdb.connect()
con.execute("SET TimeZone='UTC'")


def q(sql):
    return con.execute(sql).fetchall()


for sym in ("SPY", "IWM"):
    con.execute(f"""
    CREATE OR REPLACE TABLE m_{sym} AS
    WITH x AS (
      SELECT CAST(timezone('America/New_York', ts) AS DATE) AS date,
             CAST(extract('hour' FROM timezone('America/New_York', ts)) * 60 + extract('minute' FROM timezone('America/New_York', ts)) AS INTEGER) AS mod,
             open, high, low, close
      FROM read_parquet('{D}/alpaca/{sym}/underlying/trade/1m/v1/data/*.parquet')),
    s AS (SELECT date FROM x WHERE mod >= 570 AND mod < 960 GROUP BY date HAVING min(mod) = 570 AND max(mod) IN (959, 779))
    SELECT x.*, CAST(floor((mod - 570) / 30) AS INTEGER) AS bk FROM x JOIN s USING (date) WHERE mod >= 570 AND mod < 960
    """)

print("IWM low time (first bar holding the session low):")
rows = q("""
WITH lo AS (SELECT date, min(low) AS l FROM m_IWM GROUP BY date),
     t AS (SELECT m.date, min(m.mod) AS t FROM m_IWM m JOIN lo USING (date) WHERE m.low = lo.l GROUP BY m.date)
SELECT CAST(floor((t - 570) / 30) AS INTEGER) AS bk, count(*) AS k, round(100.0 * count(*) / (SELECT count(*) FROM t), 2) AS share FROM t GROUP BY 1 ORDER BY 1
""")
print("  sessions:", sum(r[1] for r in rows))
for r in rows:
    print("   part", r[0], r[1], r[2])

con.execute("""
CREATE OR REPLACE TABLE p AS
SELECT date, bk, arg_min(open, mod) AS first_open, arg_max(close, mod) AS last_close, max(high) AS hi, min(low) AS lo
FROM m_SPY GROUP BY date, bk
""")
for name, expr in (("abs move", "abs(last_close / first_open - 1) * 100"), ("range", "(hi - lo) / first_open * 100")):
    print(f"SPY {name} by part:")
    for r in q(f"SELECT bk, round(avg({expr}), 4), count(*) FROM p GROUP BY bk ORDER BY bk"):
        print("   part", r[0], r[1], r[2])
    print("   sessions, and sessions where part 0 holds the largest value:",
          q(f"""WITH v AS (SELECT date, bk, {expr} AS x FROM p), mx AS (SELECT date, max(x) AS m FROM v GROUP BY date),
                     b AS (SELECT v.date, min(v.bk) AS best FROM v JOIN mx USING (date) WHERE v.x = mx.m GROUP BY v.date)
                SELECT count(*), sum((best = 0)::INT), round(100.0 * sum((best = 0)::INT) / count(*), 2) FROM b"""))

print("SPY weekday (since 2015-01-01):")
con.execute(f"""
CREATE OR REPLACE TABLE d AS
SELECT date, open, close, lag(close) OVER (ORDER BY date) AS pc FROM read_parquet('{D}/yahoo/SPY/underlying/trade/1d/v1/data/daily.parquet')
""")
con.execute("""
CREATE OR REPLACE TABLE so AS SELECT date, arg_min(open, mod) AS m_open FROM m_SPY GROUP BY date
""")
for r in q("""
SELECT dayname(d.date), count(*), round(avg(d.close / d.pc - 1) * 100, 4), sum((d.close > d.pc)::INT), round(100.0 * sum((d.close > d.pc)::INT) / count(*), 2),
       round(avg(d.close / coalesce(so.m_open, d.open) - 1) * 100, 4)
FROM d LEFT JOIN so USING (date) WHERE d.pc IS NOT NULL AND d.date >= DATE '2015-01-01' GROUP BY 1, isodow(d.date) ORDER BY isodow(d.date)
"""):
    print("  ", r)
print("   all:", q("SELECT count(*), round(avg(close / pc - 1) * 100, 5), min(date), max(date) FROM d WHERE pc IS NOT NULL AND date >= DATE '2015-01-01'"))
