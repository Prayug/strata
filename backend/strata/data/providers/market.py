"""The provider marketplace: every legitimate source of historical options data that was
found, what each one ADVERTISES, and what Strata has actually VERIFIED from this machine.

The two are kept apart on purpose. `ENTRIES` is a researched catalogue: coverage, prices
and limits were read from each provider's own pages on the date in `CHECKED` (the page is
linked per entry) and are claims, not measurements. `listing()` joins it with the live
checks in `bench.py`. An integration counts as verified only after a real request from
this installation succeeded; a provider that needs credentials nobody has entered is
"unverified" however complete its adapter and its mocked tests are, and an import layout
that was only exercised on a synthetic file says exactly that.

`estimate` answers "what would it cost to obtain only the observations this strategy
needs?" for every source: requests, time and dollars where the provider publishes a price,
and "the provider quotes this exactly before you confirm" where it does not. Strata never
starts a paid request on its own.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

CHECKED = "2026-11-25"
PRIORITY = ["SPY", "QQQ", "IWM", "SPX"]


def _e(id_, name, kind, **kw):
    return {"id": id_, "name": name, "kind": kind, **kw}


# status of the integration: built-in (Strata downloads it), adapter (code exists, needs a key), import (bring the vendor's
# files), none (not usable for backtests, and why). `tested` says what the integration was actually exercised on.
ENTRIES = [
    _e("github_eod", "End-of-day option archive (public mirror)", "community dataset", source="github_eod", url="https://github.com/anahatsingh-ui/options-dataset-hist", account="none", credentials=[],
       data="End-of-day chains: bid/ask, sizes, volume, open interest, implied volatility and greeks, one snapshot per contract per session (16:00 ET)", resolution="end of day", history="2008-01 to 2025-12",
       symbols={"SPY": "2008-2025", "QQQ": "2011-2025", "IWM": "2008-2025", "SPX": "not included"}, pricing={"model": "free", "summary": "Free. MIT licence (c) 2025 Philipp D. Dubach; this is a preservation mirror of the original repository, which is no longer online."},
       limits=["One snapshot per session: cannot price intraday entries, 0DTE or intraday exits.", "About 1.7 GB in total (100-250 MB per symbol-year); Strata downloads only the years a run needs and asks before a large download.",
               "A static dataset: it ends in December 2025 and will not be extended by its author."], integration="built-in", tested="live: repository listing and a full download and normalisation of SPY years"),
    _e("dolthub", "DoltHub post-no-preference/options", "community dataset", source="dolthub", url="https://www.dolthub.com/repositories/post-no-preference/options", account="none", credentials=[],
       data="End-of-day chains with bid/ask and greeks", resolution="end of day", history="2019 to the present, updated daily", symbols={"SPY": "2019 onward, about three expirations per date", "QQQ": "same", "IWM": "same", "SPX": "not included"},
       pricing={"model": "free", "summary": "Free public database (SQL API)."}, limits=["About three expirations per date, so short-dated and 0DTE contracts are usually absent.", "End of day only."],
       integration="built-in", tested="live: latest date and one chain"),
    _e("yahoo_options", "Yahoo Finance option trade bars", "public endpoint", source="yahoo_options", url="https://finance.yahoo.com/quote/SPY/options", account="none", credentials=[],
       data="1-minute option TRADE bars per contract (no bid/ask)", resolution="1 minute", history="only contracts still listed or expired within about a day; about 7 days of 1-minute bars",
       symbols={"SPY": "rolling window only", "QQQ": "rolling window only", "IWM": "rolling window only", "SPX": "listed SPXW contracts, rolling window only (not wired into the strike grid yet)"},
       pricing={"model": "free", "summary": "Free, no key. Unofficial endpoint with no service guarantee."}, limits=["History cannot be requested later: a contract's bars disappear about a day after it expires, so they must be collected while they exist (the collector does this).",
               "Trade prints only: the bid/ask at those moments is not observed."], integration="built-in", tested="live: recent at-the-money contracts and an expired-contract depth check"),
    _e("cboe_delayed", "Cboe delayed quotes and index history", "exchange", source="cboe", url="https://www.cboe.com/delayed_quotes/spx/quote_table", account="none", credentials=[],
       data="VIX and index daily history (used for volatility conditions); delayed option chain snapshots on the website", resolution="daily (index history)", history="VIX from 1990",
       symbols={"SPY": "-", "QQQ": "-", "IWM": "-", "SPX": "delayed chain snapshot on the website only; no history"}, pricing={"model": "free", "summary": "Free index history files."},
       limits=["No historical option prices: the delayed chain is a current snapshot.", "Strata reads the index history files only."], integration="built-in", tested="live: VIX history download"),
    _e("alpaca", "Alpaca Market Data", "broker API", source="alpaca", url="https://docs.alpaca.markets/docs/historical-option-data", account="free signup (paper account is enough)", credentials=["ALPACA_API_KEY", "ALPACA_API_SECRET"],
       data="1-minute option TRADE bars, trades and (with a subscription) OPRA quotes; expired contracts included; 1-minute stock bars since 2016", resolution="1 minute", history="options since February 2024",
       symbols={"SPY": "Feb 2024 onward", "QQQ": "Feb 2024 onward", "IWM": "Feb 2024 onward", "SPX": "index options are not offered"},
       pricing={"model": "free key", "summary": "Basic plan free: historical bars are built from OPRA but the latest 15 minutes are withheld, 200 requests a minute. Algo Trader Plus $99 a month: OPRA feed and quotes, 10,000 requests a minute.",
                "tiers": [{"name": "Basic", "price": 0, "history": "Feb 2024 onward", "what": "option trade bars (not the latest 15 minutes); indicative feed for live data"}, {"name": "Algo Trader Plus", "price": 99, "history": "Feb 2024 onward", "what": "OPRA quotes and trades"}],
                "source_url": "https://docs.alpaca.markets/docs/about-market-data-api"},
       limits=["History starts in February 2024.", "Free plan: no historical bid/ask quotes (trade bars only).", "An OPRA agreement must be signed in the account for the paid feed."],
       integration="built-in", tested="adapter and pagination tested on recorded response shapes; live access is checked by the guided connection below"),
    _e("massive", "Massive (formerly Polygon.io)", "data API", source="massive", url="https://massive.com/options", account="free signup", credentials=["POLYGON_API_KEY"],
       data="Per-contract minute aggregates (TRADE bars), contract reference data including expired contracts; trades on Developer; quotes on Advanced; flat files on paid plans", resolution="1 minute (aggregates)",
       history="Basic and Starter: 2 years. Developer: 4 years. Advanced: all history (the site advertises 7+ years).", symbols={"SPY": "per plan", "QQQ": "per plan", "IWM": "per plan", "SPX": "index options (I:SPX underlying) on the Indices/Options plans; not confirmed"},
       pricing={"model": "free key", "summary": "Options Basic $0: all US option tickers, 5 API calls a minute, 2 years, end-of-day delay, minute aggregates. Starter $29, Developer $79, Advanced $199 a month (about 20% less billed annually).",
                "tiers": [{"name": "Basic", "price": 0, "history": "2 years", "what": "minute aggregates, 5 calls a minute"}, {"name": "Starter", "price": 29, "history": "2 years", "what": "unlimited calls, flat files, 15-minute delayed"},
                          {"name": "Developer", "price": 79, "history": "4 years", "what": "adds trades"}, {"name": "Advanced", "price": 199, "history": "all", "what": "adds quotes, real time"}], "source_url": "https://massive.com/options"},
       limits=["5 requests a minute on the free plan: a selective download of a few hundred contracts takes about an hour.", "Trade aggregates carry no bid/ask; quotes need the Advanced plan.",
               "Third-party summaries disagree on the history depth of the paid tiers; the provider's own page was used."], integration="adapter", tested="adapter tested on documented response shapes only; never run against the live service (no key)"),
    _e("alphavantage", "Alpha Vantage historical options", "data API", source="alphavantage", url="https://www.alphavantage.co/documentation/#historical-options", account="paid key (the demo key serves IBM only)", credentials=["ALPHAVANTAGE_API_KEY"],
       data="End-of-day chains by date: bid/ask, sizes, volume, open interest, implied volatility and greeks", resolution="end of day", history="any session since 2008-01-01 (15+ years advertised)",
       symbols={"SPY": "2008 onward with a premium key", "QQQ": "2008 onward with a premium key", "IWM": "2008 onward with a premium key", "SPX": "not stated in the documentation"},
       pricing={"model": "subscription", "summary": "The endpoint is marked Premium. Plans from $49.99 a month (75 requests a minute); no daily cap on paid plans.",
                "tiers": [{"name": "75 requests/min", "price": 49.99, "history": "2008 onward", "what": "one request returns one symbol's full chain for one session"}], "source_url": "https://www.alphavantage.co/premium/"},
       limits=["One snapshot per session (end of day).", "One request per symbol-session: a year of one symbol is about 252 requests.", "The key can only be sent as a URL parameter; Strata sends it over HTTPS and removes it from every log and error message."],
       integration="adapter", tested="endpoint and parser verified live with the public demo key (IBM); other symbols need your own key"),
    _e("databento", "Databento OPRA", "data API (usage-based)", source="databento", url="https://databento.com/datasets/OPRA.PILLAR", account="signup; $125 of free historical credit for new accounts (expires after 6 months)", credentials=["DATABENTO_API_KEY"],
       data="The full OPRA feed: trades, 1-minute OHLCV, consolidated bid/ask sampled each minute (CBBO-1m), instrument definitions and statistics; NBBO at each trade (TCBBO) from March 2023", resolution="tick to 1 minute",
       history="Trades, OHLCV, definitions and CBBO-1m from 2013-04-01; TCBBO from 2023-03-28", symbols={"SPY": "2013-04 onward", "QQQ": "2013-04 onward", "IWM": "2013-04 onward", "SPX": "2013-04 onward (SPX and SPXW are in OPRA)"},
       pricing={"model": "usage", "summary": "Pay per byte of historical data, quoted before download; the site lists OPRA historical data \"from $0.04/GB\" and the per-schema rate is shown in its catalogue. Metadata and cost quotes are free.",
                "source_url": "https://databento.com/docs/api-historical/basics/metered-pricing"},
       limits=["The only source found with intraday bid/ask history back to 2013 at retail prices.", "Many older OPRA days are marked degraded (minor packet loss) by the provider.", "Cost depends on how much is requested: a selective request is cheap, a whole chain is not."],
       integration="adapter", tested="adapter written against the documented API; never run against the live service (no key)"),
    _e("thetadata", "ThetaData", "data API (local terminal)", source="thetadata", url="https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html", account="signup; their terminal application must be running", credentials=[],
       data="End-of-day on the free tier; 1-minute quotes and OHLC on Value; tick trades and quotes with greeks on Standard and Pro", resolution="end of day to tick",
       history="Free: from 2023-06-01 (1 day delayed, 30 requests a minute). Value: from 2020-01-01. Standard: from 2016-01-01. Pro: from 2012-06-01.",
       symbols={"SPY": "per tier", "QQQ": "per tier", "IWM": "per tier", "SPX": "per tier (index options are covered)"},
       pricing={"model": "subscription", "summary": "Free tier (end of day). Value $40, Standard $80, Pro $160 a month.",
                "tiers": [{"name": "Free", "price": 0, "history": "from 2023-06-01", "what": "end of day"}, {"name": "Value", "price": 40, "history": "from 2020-01-01", "what": "1-minute quotes and OHLC"},
                          {"name": "Standard", "price": 80, "history": "from 2016-01-01", "what": "tick quotes and trades"}, {"name": "Pro", "price": 160, "history": "from 2012-06-01", "what": "tick, greeks"}],
                "source_url": "https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html"},
       limits=["Requests go through a local terminal program, not a hosted API.", "Individual-use pricing; commercial use is a separate offering."], integration="adapter", tested="adapter written against the documented API; never run against the live service"),
    _e("orats", "ORATS", "data API / files", source="orats", url="https://orats.com/one-minute-data", account="paid", credentials=["ORATS_API_KEY"], data="1-minute option data since October 2020; near-end-of-day history since 2007; smoothed volatility surfaces",
       resolution="1 minute and end of day", history="1-minute from 2020-10; near end of day from 2007", symbols={"SPY": "yes", "QQQ": "yes", "IWM": "yes", "SPX": "yes"},
       pricing={"model": "subscription or one-time", "summary": "API subscriptions from about $99 a month; the full 1-minute history is sold as a one-time delivery listed at $1,500 (about 28 TB, with checksum manifests).", "source_url": "https://orats.com/hard-drive-delivery"},
       limits=["The complete intraday set is far larger than a laptop holds."], integration="adapter", tested="adapter written against the documented API; never run against the live service"),
    _e("cboe_datashop", "Cboe DataShop", "exchange files", source=None, url="https://datashop.cboe.com/option-quotes-intervals", account="paid per order", credentials=[],
       data="Option quote intervals (NBBO, sizes, OHLC and volume per interval, optional greeks); end-of-day quotes; trades", resolution="1 minute or any N-minute interval; end of day", history="2004 to the present",
       symbols={"SPY": "2004 onward", "QQQ": "2004 onward", "IWM": "2004 onward", "SPX": "2012 onward; SPX bid/ask requires a Cboe Global Indices licence (from $1,000 a month)"},
       pricing={"model": "per order", "summary": "Priced per symbol and period at checkout; the product pages do not publish a price list. Subscriptions for ongoing delivery exist.", "source_url": "https://datashop.cboe.com/options-intervals"},
       limits=["The reference source for interval bid/ask history, and the most expensive of the retail routes.", "Purchased files may not be redistributed."], integration="import", import_format="cboe_interval / cboe_eod",
       tested="import layout tested on a synthetic file with the documented columns; not tested on a purchased file"),
    _e("historicaloptiondata", "historicaloptiondata.com", "bulk files", source=None, url="https://historicaloptiondata.com/", account="paid, one-time", credentials=[], data="End-of-day chains for all US equity options (also 1-minute data)",
       resolution="end of day (1 minute sold separately)", history="2002 onward (SPX from 1990)", symbols={"SPY": "from 2005", "QQQ": "yes", "IWM": "yes", "SPX": "from 1990"},
       pricing={"model": "one-time", "summary": "One-time purchases; the SPY option history from 2005 is listed at $425.", "source_url": "https://historicaloptiondata.com/"}, limits=["Delivered as ZIP archives of CSV files."],
       integration="import", import_format="historicaloptiondata", tested="import layout tested on a synthetic file with the documented columns; not tested on a purchased file"),
    _e("optionsdx", "OptionsDX", "bulk files", source=None, url="https://www.optionsdx.com/", account="free signup", credentials=[], data="End-of-day and intraday option chains as monthly CSV files", resolution="end of day and intraday snapshots",
       history="varies by symbol", symbols={"SPY": "listed", "QQQ": "listed", "IWM": "not confirmed", "SPX": "listed"}, pricing={"model": "free / low cost", "summary": "Older years are offered as free downloads, others at low per-file prices; terms were not confirmed for this report."},
       limits=["Manual downloads. Check the site's terms for your use; they were not verified here."], integration="import", import_format="optionsdx", tested="import layout tested on a synthetic file with the vendor's column names; not tested on a downloaded file"),
    _e("algoseek_aws", "Algoseek free samples on AWS Data Exchange", "sample files", source=None, url="https://aws.amazon.com/marketplace/pp/prodview-sk2vtzeo5noje", account="AWS account (free subscription to the listing)", credentials=[],
       data="US equity options trade+quote 1-minute bars built from the full OPRA feed (bid/ask and trades per minute)", resolution="1 minute", history="one month of 2020 (sample)",
       symbols={"SPY": "sample month only", "QQQ": "sample month only", "IWM": "sample month only", "SPX": "index options are a separate listing"},
       pricing={"model": "free sample", "summary": "The listing is offered free of charge; the full history is a commercial product.", "source_url": "https://aws.amazon.com/marketplace/pp/prodview-sk2vtzeo5noje"},
       limits=["A single month: enough to test an intraday bid/ask workflow, not to backtest.", "Requires an AWS account; Strata does not download it for you."], integration="import", import_format="generic option quotes (column mapping)",
       tested="not tested: no file from this listing was obtained"),
    _e("wrds_optionmetrics", "OptionMetrics IvyDB via WRDS", "academic", source=None, url="https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/", account="university subscription", credentials=[],
       data="End-of-day best bid/offer, volume, open interest, implied volatility and greeks for all US listed equity and index options", resolution="end of day", history="January 1996 onward (the academic copy usually lags by about a year)",
       symbols={"SPY": "1996 onward", "QQQ": "1999 onward", "IWM": "2000 onward", "SPX": "1996 onward"}, pricing={"model": "institutional", "summary": "No individual price: access comes with a university's WRDS subscription, for research and teaching only."},
       limits=["Available only to students and staff of subscribing institutions; non-commercial use.", "End of day only."], integration="import", import_format="wrds_optionmetrics",
       tested="import layout tested on a synthetic file with the documented WRDS column names; not tested on a real export"),
    _e("firstrate", "FirstRate Data", "bulk files", source=None, url="https://firstratedata.com/options-data", account="paid, one-time", credentials=[], data="End-of-day option chains; intraday stock and ETF bars", resolution="end of day (options)", history="options from January 2010",
       symbols={"SPY": "from 2010", "QQQ": "from 2010", "IWM": "from 2010", "SPX": "listed among indices"}, pricing={"model": "one-time", "summary": "Perpetual licence per bundle; prices are shown at checkout."}, limits=["Purchased files."],
       integration="import", import_format="generic option quotes (column mapping)", tested="not tested on a purchased file"),
    _e("marketdata_app", "MarketData.app", "data API", source=None, url="https://www.marketdata.app/pricing/", account="free signup", credentials=[], data="Historical option quotes and chains (end of day)", resolution="end of day",
       history="Free: 1 year. Starter: 5 years. Trader and above: no limit.", symbols={"SPY": "per plan", "QQQ": "per plan", "IWM": "per plan", "SPX": "per plan"},
       pricing={"model": "subscription", "summary": "Free: 100 credits a day, 1 year of history. Starter $30 a month ($12 billed annually): 10,000 credits a day, 5 years. Trader $75 ($30 annually): unlimited history.", "source_url": "https://www.marketdata.app/pricing/"},
       limits=["One credit per option symbol returned: a full SPY chain uses thousands, so the free tier covers a handful of contracts a day."], integration="none", tested="no integration"),
    _e("eodhd", "EODHD US options", "data API", source=None, url="https://eodhd.com/lp/us-stock-options-api", account="paid", credentials=[], data="End-of-day option chains with greeks for 6,000+ US symbols", resolution="end of day", history="not stated on the product page",
       symbols={"SPY": "listed", "QQQ": "listed", "IWM": "listed", "SPX": "not stated"}, pricing={"model": "subscription", "summary": "$29.99 a month for the options API.", "source_url": "https://eodhd.com/lp/us-stock-options-api"},
       limits=["History depth is not published."], integration="none", tested="no integration"),
    _e("tradier", "Tradier brokerage API", "broker API", source=None, url="https://documentation.tradier.com/", account="brokerage or sandbox account", credentials=[], data="Current option chains and quotes; price history for symbols still listed",
       resolution="end of day and intraday for listed symbols", history="none for expired options (stated in their documentation)", symbols={"SPY": "listed contracts only", "QQQ": "listed contracts only", "IWM": "listed contracts only", "SPX": "listed contracts only"},
       pricing={"model": "free with account", "summary": "Market data comes with the account."}, limits=["Expired contracts are not kept, so it cannot supply a backtest's history."], integration="none", tested="no integration (cannot serve history)"),
    _e("ibkr", "Interactive Brokers API", "broker API", source=None, url="https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/", account="brokerage account with market data subscriptions", credentials=[],
       data="Historical bars for currently listed option contracts, one contract per request", resolution="intraday bars for listed contracts", history="none for expired options (stated in their documentation); no end-of-day option bars through the API",
       symbols={"SPY": "listed contracts only", "QQQ": "listed contracts only", "IWM": "listed contracts only", "SPX": "listed contracts only"}, pricing={"model": "subscription", "summary": "Requires the account's OPRA market data subscription."},
       limits=["No history for expired options, so it cannot supply a backtest."], integration="none", tested="no integration (cannot serve history)"),
    _e("quantconnect", "QuantConnect (Algoseek data)", "hosted research", source=None, url="https://www.quantconnect.com/datasets/algoseek-us-equity-options", account="free signup", credentials=[],
       data="Minute option quotes and trades inside their cloud research environment", resolution="1 minute", history="2012 onward (equity options)", symbols={"SPY": "in their cloud", "QQQ": "in their cloud", "IWM": "in their cloud", "SPX": "in their cloud"},
       pricing={"model": "free in the cloud", "summary": "Usable for research in their environment; the data cannot be exported."}, limits=["The data stays on their platform, so Strata cannot use it."], integration="none", tested="no integration (data cannot be exported)"),
]

# what a selective download costs in requests and time, per source (dollars only where the provider publishes them)
COST = {
    "alpaca": {"per_minute": 200, "batch": 100, "price": 0.0, "window": lambda today: (date(2024, 2, 1), today), "what": "trade bars"},
    "massive": {"per_minute": 5, "batch": 1, "price": 0.0, "window": lambda today: (today - timedelta(days=730), today - timedelta(days=1)), "what": "trade aggregates", "merge_sessions": True},
    "yahoo_options": {"per_minute": 180, "batch": 1, "price": 0.0, "window": lambda today: (today - timedelta(days=2), today), "what": "trade bars"},
    "alphavantage": {"per_minute": 75, "batch": None, "price": 49.99, "window": lambda today: (date(2008, 1, 2), today - timedelta(days=1)), "what": "end-of-day chains", "per_session": True},
    "github_eod": {"per_minute": None, "batch": None, "price": 0.0, "window": lambda today: (date(2008, 1, 2), date(2025, 12, 12)), "what": "end-of-day chains", "per_year_mb": 180},
    "thetadata": {"per_minute": None, "batch": None, "price": 40.0, "window": lambda today: (date(2020, 1, 1), today), "what": "1-minute quotes"},
    "databento": {"per_minute": None, "batch": None, "price": None, "window": lambda today: (date(2013, 4, 1), today - timedelta(days=1)), "what": "1-minute bid/ask (CBBO-1m)", "record_bytes": 80, "floor_per_gb": 0.04},
}


def listing(store=None) -> dict:
    """The marketplace rows: the researched claims joined with what was verified here."""
    from strata.api.settings import get_secret
    from strata.data.providers import bench as B
    from strata.data.providers import sources as S
    res = B.saved()
    rows = []
    for e in ENTRIES:
        row = {k: v for k, v in e.items()}
        src_name = e.get("source") if e.get("source") in S.SOURCES else None
        b = res.get(e["id"]) or (res.get(src_name) if src_name else None)
        row["credentials_configured"] = [c for c in e["credentials"] if get_secret(c)]
        row["ready"] = bool(src_name) and len(row["credentials_configured"]) == len(e["credentials"])
        checkable = src_name in B.BENCHABLE or e["id"] in B.BENCHABLE
        if e["integration"] == "adapter" and not checkable:
            ver = {"status": "unverified", "detail": "No live check exists for this adapter yet. " + e["tested"][0].upper() + e["tested"][1:] + "."}
        elif e["integration"] == "none":
            ver = {"status": "no integration", "detail": e["tested"]}
        elif e["integration"] == "import":
            ver = {"status": "import only", "detail": e["tested"]}
        elif b:
            ver = {"status": b.get("status") or ("verified" if b.get("verified") else "failed"), "checked_at": b.get("measured_at"), "detail": b.get("reason") or "; ".join(n for n in (b.get("notes") or [])[:2]),
                   "steps": [{"name": s["name"], "ok": s["ok"], "detail": s.get("error") or s.get("dataset") or ""} for s in b.get("steps") or []], "entitlements": b.get("entitlements") or [], "depth": b.get("historical_depth") or []}
        elif e["credentials"] and not row["ready"]:
            ver = {"status": "unverified", "detail": "No credentials are configured, so no live request has been made. " + e["tested"] + "."}
        else:
            ver = {"status": "not checked yet", "detail": "A live check is available and has not been run on this installation."}
        row["verification"] = ver
        row["can_check"] = checkable
        row["check_name"] = e["id"] if e["id"] in B.BENCHABLE else src_name
        row["advertised"] = {"data": e["data"], "resolution": e["resolution"], "history": e["history"], "symbols": e["symbols"]}
        rows.append(row)
    order = {"verified": 0, "partly verified": 1, "parser verified (demo symbol only)": 2, "not checked yet": 3, "unverified": 4, "failed": 5, "import only": 6, "no integration": 7}
    rows.sort(key=lambda r: (order.get(r["verification"]["status"], 9), r["name"]))
    return {"checked": CHECKED, "priority_symbols": PRIORITY, "sources": rows,
            "note": "Coverage, prices and limits are what each provider advertises, read from its own pages on the date shown. \"Verified\" means a real request from this installation succeeded; nothing else earns that word.",
            "summary": {"verified": sum(1 for r in rows if r["verification"]["status"] in ("verified", "partly verified")), "unverified": sum(1 for r in rows if r["verification"]["status"] in ("unverified", "not checked yet", "parser verified (demo symbol only)")),
                        "import_only": sum(1 for r in rows if r["integration"] == "import"), "no_integration": sum(1 for r in rows if r["integration"] == "none")}}


# ----------------------------------------------------------------------------- incremental cost of exactly what a strategy needs
def needs_of(config: dict, store=None, max_signals: int = 4000) -> dict:
    """What observed option data a backtest configuration needs: the sessions on which its
    entry rule fires (found by scanning the underlying only), how long positions are held,
    and how many contracts the strike band around each signal contains."""
    from strata.data import planner as PL
    from strata.data.store import MarketStore
    from strata.engine.models import BacktestConfig
    from strata.engine.scan import scan_signals
    store = store or MarketStore()
    cfg = BacktestConfig.model_validate(config)
    strat = cfg.strategies[0]
    sigs = scan_signals(cfg, store)
    signals = sigs.get("signals", [])[:max_signals]
    hold = PL.hold_sessions(strat)
    legs = [l for l in strat.legs if l.kind == "option"]
    per_side = 2 * 6 + 1  # the planner's default strike band: about six strikes either side of each leg's target
    contracts = len(legs) * per_side
    days = sorted({s["date"] if isinstance(s.get("date"), str) else str(s.get("date")) for s in signals})
    mode = cfg.pricing.mode
    try:
        have_q = {str(d) for d in store.option_days(strat.symbol, "quote")}
    except Exception:  # noqa: BLE001
        have_q = set()
    try:
        idx = store.option_bar_index(strat.symbol)
        have_t = {str(x)[:10] for x in idx["date"].to_list()} if not idx.is_empty() and "date" in idx.columns else set()
    except Exception:  # noqa: BLE001
        have_t = set()
    return {"symbol": strat.symbol, "signals": len(signals), "signal_sessions": len(days), "first": days[0] if days else None, "last": days[-1] if days else None, "hold_sessions": hold, "option_legs": len(legs),
            "contracts_per_signal": contracts, "sessions_per_signal": hold + 1, "days": days, "pricing": mode, "sessions_with_quotes": sum(1 for d in days if d in have_q), "sessions_with_trades": sum(1 for d in days if d in have_t),
            "excluded_sessions": len({d for v in (sigs.get("excluded") or {}).values() for d in v}), "sessions_scanned": sigs.get("sessions"), "intraday": any(s.get("minute", 570) not in (570, 960) for s in signals)}


def estimate(need: dict, today: date | None = None) -> dict:
    """Per source: can it supply these observations, in how many requests, how long, and at what price."""
    from strata.api.settings import get_secret
    today = today or date.today()
    days = [date.fromisoformat(d) for d in need.get("days", [])]
    by_id = {e["id"]: e for e in ENTRIES}
    rows = []
    for sid, c in COST.items():
        e = by_id[sid]
        a, b = c["window"](today)
        inside = [d for d in days if a <= d <= b]
        row = {"id": sid, "name": e["name"], "what": c["what"], "window": [a.isoformat(), b.isoformat()], "signal_sessions": len(days), "covered_sessions": len(inside), "coverage_pct": round(len(inside) / len(days) * 100, 1) if days else None,
               "requests": None, "minutes": None, "dollars": None, "dollars_note": "", "notes": [], "needs": e["credentials"], "ready": all(get_secret(k) for k in e["credentials"]), "integration": e["integration"]}
        eod = c["what"].startswith("end-of-day")
        if eod and need.get("intraday"):
            row["notes"].append("End-of-day snapshots cannot price this strategy's intraday entries: only entries and exits at the close are testable on it.")
        pairs = len(inside) * need["contracts_per_signal"] * need["sessions_per_signal"]
        if sid == "github_eod":
            years = sorted({d.year for d in inside})
            row.update(requests=len(years), megabytes=len(years) * c["per_year_mb"], minutes=round(len(years) * 0.6, 1), dollars=0.0)
            row["notes"].append(f"Whole year files are downloaded: {len(years)} file{'s' if len(years) != 1 else ''} of about {c['per_year_mb']} MB cover every session of those years, not only the signal sessions.")
        elif c.get("per_session"):
            n = len({d + timedelta(days=k) for d in inside for k in range(need["sessions_per_signal"])})
            row.update(requests=n, minutes=round(n / c["per_minute"], 1), dollars=c["price"], dollars_note="one month of the entry plan; the requests themselves are not charged")
        elif c.get("per_minute"):
            reqs = len(inside) * need["contracts_per_signal"] if c.get("merge_sessions") else math.ceil(pairs / max(1, c["batch"]))
            row.update(requests=reqs, minutes=round(reqs / c["per_minute"], 1), dollars=c["price"], contract_sessions=pairs)
            if c["per_minute"] <= 5 and reqs > 300:
                row["notes"].append(f"At 5 requests a minute this takes about {reqs / 5 / 60:.1f} hours; a paid plan removes the limit.")
        elif sid == "databento":
            recs = pairs * 390
            gb = recs * c["record_bytes"] / 1e9
            row.update(records=recs, gigabytes=round(gb, 4), dollars=None, dollars_floor=round(gb * c["floor_per_gb"], 2), contract_sessions=pairs,
                       dollars_note="Not estimated: the per-schema rate is only shown in the provider's catalogue. The lowest published OPRA rate would make this at least the floor shown; the provider returns the exact price, free of charge, before any download.")
        elif sid == "thetadata":
            row.update(dollars=c["price"], dollars_note="one month of the Value tier (1-minute quotes from 2020); earlier dates need Standard ($80) or Pro ($160)", contract_sessions=pairs)
            if inside and min(inside) < date(2020, 1, 1):
                row["notes"].append("Some signal sessions are before 2020, which the Value tier does not reach.")
        if not inside and days:
            row["notes"].append("None of the signal sessions fall inside this source's history.")
        rows.append(row)
    rows.sort(key=lambda r: (-(r["coverage_pct"] or 0), r["dollars"] if r["dollars"] is not None else 1e9))
    return {"need": {k: v for k, v in need.items() if k != "days"}, "sources": rows,
            "notes": ["Signals are found by scanning the underlying only, so no option data is needed to produce this estimate.",
                      "Nothing here is started automatically. A free source is downloaded when you run the backtest (large downloads ask first); a paid one is never called without your approval of its quoted cost.",
                      "Dollar figures are the providers' published subscription prices on " + CHECKED + ", not charges Strata can see or control."]}


# ----------------------------------------------------------------------------- guided Alpaca connection
def alpaca_validation(store=None, progress=None, should_cancel=None, symbol: str = "SPY") -> dict:
    """Connect, discover what the account can really read, fetch an expired SPY option, store
    its trade bars and run a small backtest on them. Every step reports what happened; with
    no credentials every step is "not run" and nothing claims success."""
    from strata.core import calendar as cal
    from strata.data.providers import bench as B
    from strata.data.providers import sources as S
    from strata.data.store import MarketStore
    store = store or MarketStore()
    src = S.Alpaca(store)
    steps: list[dict] = []
    p = progress or (lambda f, m: None)

    def add(sid, title, status, detail, **kw):
        steps.append({"id": sid, "title": title, "status": status, "detail": detail, **kw})

    names = [("credentials", "Credentials stored on this computer"), ("entitlements", "What this account can read"), ("contract", f"An expired {symbol} option contract"), ("bars", "Its historical 1-minute trade bars"), ("backtest", "A small backtest on those observed prices")]
    ok, why = src.available()
    if not ok:
        add("credentials", names[0][1], "missing", "No Alpaca key is stored. Create a free Alpaca account (a paper account is enough), then enter the key ID and secret in Settings; they are kept in this computer's credential store and never sent anywhere but Alpaca.")
        for sid, title in names[1:]:
            add(sid, title, "not_run", "Not run: there are no credentials to make the request with.")
        out = {"provider": "alpaca", "ok": False, "verified": False, "steps": steps, "summary": "Alpaca is not connected. Nothing was requested and nothing is claimed to work."}
        B._save("alpaca_validation", {**out, "measured_at": B.datetime.now(B.timezone.utc).isoformat()})
        return out
    add("credentials", names[0][1], "ok", "A key ID and secret are stored.")
    p(0.1, "checking entitlements")
    bench = B.benchmark_source("alpaca", symbol=symbol)
    ent = bench.get("entitlements") or []
    granted = [x["what"] for x in ent if x.get("granted")]
    if not bench.get("verified"):
        add("entitlements", names[1][1], "failed", "Alpaca rejected every request: " + "; ".join(str(s.get("error")) for s in bench.get("steps", []) if not s.get("ok"))[:300], entitlements=ent)
        for sid, title in names[2:]:
            add(sid, title, "not_run", "Not run: the account could not be read.")
        out = {"provider": "alpaca", "ok": False, "verified": False, "steps": steps, "summary": "The stored Alpaca credentials were not accepted."}
        B._save("alpaca_validation", {**out, "measured_at": bench.get("measured_at")})
        return out
    add("entitlements", names[1][1], "ok", f"{len(granted)} of {len(ent)} capabilities are available: " + "; ".join(granted) + ".", entitlements=ent, rate_limits=bench.get("rate_limits"))
    # an at-the-money call that expired about a month ago, chosen from the price stored for that date
    today = B.datetime.now(cal.ET).date()
    exp = [d for d in cal.trading_days(today - timedelta(days=45), today - timedelta(days=25)) if d.weekday() == 4][-1:] or [cal.trading_days(today - timedelta(days=45), today - timedelta(days=25))[-1]]
    exp = exp[0]
    ref = S.previous_close(store, symbol, exp)
    contract, wanted = None, {}
    try:
        if ref is None:
            raise S.SourceError(f"no daily price for {symbol} is stored around {exp}; download the free daily history first")
        p(0.4, "looking up an expired contract")
        with S._client() as client:
            cs = src.list_contracts(client, symbol, exp, exp, exp, ref * 0.99, ref * 1.01, rights=("C",))
        if not cs:
            raise S.SourceError(f"Alpaca listed no {symbol} call expiring {exp} near {ref:.0f}")
        contract = min(cs, key=lambda c: abs(c.strike - ref))
        add("contract", names[2][1], "ok", f"{contract.occ}: the {contract.strike:g} call that expired on {exp} ({symbol} closed at {ref:.2f} the session before). {contract.source}.", occ=contract.occ, expiration=str(exp), strike=contract.strike)
    except S.SourceError as e:
        add("contract", names[2][1], "failed", str(e)[:300])
    if contract is None:
        for sid, title in names[3:]:
            add(sid, title, "not_run", "Not run: no contract was retrieved.")
    else:
        try:
            p(0.6, "downloading its trade bars")
            wanted = {exp: [contract]}
            rep = src.acquire(symbol, wanted, should_cancel=should_cancel)
            idx = store.option_bar_index(symbol, provider="alpaca")
            n = int(idx.filter((idx["occ"] == contract.occ) & (idx["date"] == exp))["bars"].sum()) if not idx.is_empty() and "bars" in idx.columns else None
            if not n:
                raise S.SourceError(f"the request succeeded but returned no trade bars for {contract.occ} on {exp} (the contract may not have traded)")
            add("bars", names[3][1], "ok", f"{n} one-minute trade bars for {contract.occ} on {exp} are stored. These are trade prints: no bid/ask is included on the free plan.", bars=n, report={k: rep.get(k) for k in ("pairs_ok", "requests", "bars") if k in rep})
        except (S.SourceError, Exception) as e:  # noqa: BLE001
            add("bars", names[3][1], "failed", f"{type(e).__name__}: {e}"[:300])
        if steps[-1]["status"] != "ok":
            add("backtest", names[4][1], "not_run", "Not run: no trade bars were stored.")
        else:
            try:
                p(0.85, "running a small backtest")
                from strata.engine.engine import run_backtest
                cfg = {"start": exp.isoformat(), "end": exp.isoformat(), "pricing": {"mode": "trades"}, "data": {"options_provider": "alpaca"}, "execution": {"signal_interval_minutes": 5},
                       "strategies": [{"name": "Alpaca connection check", "symbol": symbol, "legs": [{"id": "c", "side": "buy", "right": "C", "dte": 0, "strike": {"method": "fixed", "value": contract.strike}}],
                                       "entry": {"conditions": {"kind": "group", "op": "and", "children": [{"kind": "cond", "type": "time_window", "params": {"start": "10:00", "end": "11:00"}}]}},
                                       "exits": {"profit_target": {"enabled": False}, "stop_loss": {"enabled": False}, "time_exit": {"enabled": True, "time": "15:00"}}}]}
                res = run_backtest(cfg, store=store, record_paths=False)
                if not res.trades:
                    skipped = (res.diagnostics or {}).get("skipped_signals") or []
                    raise RuntimeError("no trade was filled: " + (skipped[0]["reason"] if skipped else "no print was recent enough at the entry"))
                t = res.trades[0]
                add("backtest", names[4][1], "ok", f"One long call, entered {t['entry_time'][11:16]} ET at {t['legs'][0]['entry_price']:.2f} and exited {t['exit_time'][11:16]} ET at {t['legs'][0]['exit_price']:.2f} "
                    f"({t['exit_reason'].replace('_', ' ')}): P&L {t['pnl']:+,.2f}. Both prices are observed trade prints; nothing was modelled.", trade={k: t[k] for k in ("entry_time", "exit_time", "pnl", "exit_reason")})
            except Exception as e:  # noqa: BLE001
                add("backtest", names[4][1], "failed", f"{type(e).__name__}: {e}"[:300])
    good = all(s["status"] == "ok" for s in steps)
    out = {"provider": "alpaca", "ok": good, "verified": good, "steps": steps, "measured_at": bench.get("measured_at"),
           "summary": "Alpaca is connected and a historical option backtest ran on prices downloaded from it." if good else "Alpaca is connected, but not every step completed: see the steps for exactly what did and did not work."}
    B._save("alpaca_validation", out)
    return out
