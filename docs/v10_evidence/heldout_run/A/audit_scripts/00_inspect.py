"""Inspect the raw price files: schema, ranges, duplicates, nulls. Read-only."""
import glob
import polars as pl

DATA = "/Users/prayugsigdel/Coding/strata/data"

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

for sym in ["SPY", "QQQ", "IWM"]:
    p = f"{DATA}/market/datasets/yahoo/{sym}/underlying/trade/1d/v1/data/daily.parquet"
    d = pl.read_parquet(p)
    print("=" * 100)
    print("DAILY", sym, p)
    print(d.schema)
    print("rows", d.height, "first", d["date"].min(), "last", d["date"].max())
    print("dup dates", d.height - d["date"].n_unique())
    print("nulls", d.null_count().to_dicts())
    print("sorted", d["date"].is_sorted())
    print(d.head(3))
    print(d.tail(5))
    # sanity
    bad = d.filter((pl.col("high") < pl.col("low")) | (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
                   | (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close")))
    print("rows where OHLC inconsistent:", bad.height)
    if bad.height:
        print(bad.head(20))
    zero = d.filter((pl.col("open") <= 0) | (pl.col("close") <= 0) | (pl.col("volume") <= 0))
    print("rows with nonpositive open/close/volume:", zero.height)
    if zero.height:
        print(zero.head(20))

vix_paths = glob.glob(f"{DATA}/market/datasets/cboe/VIX/underlying/*/1d/v1/data/daily.parquet")
print("=" * 100)
print("VIX paths", vix_paths)
for p in vix_paths:
    d = pl.read_parquet(p)
    print(d.schema)
    print("rows", d.height, "first", d["date"].min(), "last", d["date"].max())
    print("dup dates", d.height - d["date"].n_unique())
    print("nulls", d.null_count().to_dicts())
    print(d.head(3))
    print(d.tail(5))

for sym in ["SPY", "QQQ", "IWM"]:
    print("=" * 100)
    print("MINUTE", sym)
    files = sorted(glob.glob(f"{DATA}/market/datasets/alpaca/{sym}/underlying/trade/1m/v1/data/*.parquet"))
    for f in files:
        m = pl.read_parquet(f)
        print(f.split("/")[-1], m.schema if f == files[0] else "", "rows", m.height, "first", m["ts"].min(), "last", m["ts"].max(),
              "dup", m.height - m["ts"].n_unique(), "nulls", sum(m.null_count().row(0)))
    m = pl.read_parquet(files[0])
    print(m.head(5))
