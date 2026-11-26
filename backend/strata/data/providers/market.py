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
       data="1-minute option TRADE bars, trades and (with a subscription) OPRA quotes; expired contracts included; 1-minute stock bars since 2016", resolution="1 minute", history="options since 2024-01-18 (measured with a paper key; 2024-02-02 is missing at the provider)",
       symbols={"SPY": "2024-01-18 onward (measured)", "QQQ": "2024-01 onward (not measured here)", "IWM": "2024-01 onward (not measured here)", "SPX": "index options are not offered"},
       pricing={"model": "free key", "summary": "Basic plan free: historical bars are built from OPRA but the latest 15 minutes are withheld, 200 requests a minute. Algo Trader Plus $99 a month: OPRA feed and quotes, 10,000 requests a minute.",
                "tiers": [{"name": "Basic", "price": 0, "history": "2024-01-18 onward (measured)", "what": "option trade bars (not the latest 15 minutes); indicative feed for live data"}, {"name": "Algo Trader Plus", "price": 99, "history": "the same history", "what": "OPRA quotes and trades"}],
                "source_url": "https://docs.alpaca.markets/docs/about-market-data-api"},
       limits=["History starts on 2024-01-18 (measured); 2024-02-02 returns nothing.", "Free plan: no historical bid/ask quotes (trade bars only).", "An OPRA agreement must be signed in the account for the paid feed."],
       integration="built-in", tested="adapter and pagination tested on recorded response shapes. On 2026-11-25 every endpoint Strata uses answered 401 without a key (they exist; only authentication is missing). "
                                      "What a key really returns is established by the access check: an expired-contract matrix across 2024-2026"),
    _e("massive", "Massive (formerly Polygon.io)", "data API", source="massive", url="https://massive.com/options", account="free signup", credentials=["POLYGON_API_KEY"],
       data="Per-contract minute aggregates (TRADE bars), contract reference data including expired contracts; trades on Developer; quotes on Advanced; flat files on paid plans", resolution="1 minute (aggregates)",
       history="Basic and Starter: 2 years. Developer: 4 years. Advanced: all history (the site advertises 7+ years).", symbols={"SPY": "per plan", "QQQ": "per plan", "IWM": "per plan", "SPX": "index options (I:SPX underlying) on the Indices/Options plans; not confirmed"},
       pricing={"model": "free key", "summary": "Options Basic $0: all US option tickers, 5 API calls a minute, 2 years, end-of-day delay, minute aggregates. Starter $29, Developer $79, Advanced $199 a month (about 20% less billed annually).",
                "tiers": [{"name": "Basic", "price": 0, "history": "2 years", "what": "minute aggregates, 5 calls a minute"}, {"name": "Starter", "price": 29, "history": "2 years", "what": "unlimited calls, flat files, 15-minute delayed"},
                          {"name": "Developer", "price": 79, "history": "4 years", "what": "adds trades"}, {"name": "Advanced", "price": 199, "history": "all", "what": "adds quotes, real time"}], "source_url": "https://massive.com/options"},
       limits=["5 requests a minute on the free plan: a selective download of a few hundred contracts takes about an hour.", "Trade aggregates carry no bid/ask; quotes need the Advanced plan, and its quote history starts 2022-03-07.",
               "Individual trades need the Developer plan. Expired contracts are listed on every plan (expired=true).",
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
       pricing={"model": "usage", "summary": "Pay per byte of historical data. The documented example rates for OPRA are $2 per GB for 1-minute bid/ask (cbbo-1m) and $280 per GB for trades and 1-minute OHLCV. "
                                             "A new account gets $125 of credit (one per team, expires after 6 months). The exact price of a request is returned free of charge before any download (metadata.get_cost).",
                "tiers": [{"name": "Free credit", "price": 0, "history": "2013-04 onward", "what": "$125 of usage: at $2 per GB, about two million contract-sessions of 1-minute bid/ask"}, {"name": "Usage", "price": None, "history": "2013-04 onward", "what": "$2 per GB (cbbo-1m), quoted per request"}],
                "source_url": "https://databento.com/docs/api-reference-historical/metadata/metadata-list-unit-prices"},
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
    _e("optionsdx", "OptionsDX", "bulk files", source=None, url="https://www.optionsdx.com/product/spy-option-chains/", account="free signup (needed at checkout, also for the free files)", credentials=[],
       data="Whole option chains as monthly CSV files: bid, ask, last, sizes, volume, greeks and implied volatility for every strike and expiration, with the underlying price at each snapshot",
       resolution="end of day, 30, 15, 5 or 1 minute snapshots", history="SPY 2010-2023; QQQ 2012-2023; VIX 2010-2023. Nothing after 2023 was listed.",
       symbols={"SPY": "2010-2023 (5-minute from 2016, 1-minute 2020-2023 on the storefront)", "QQQ": "2012-2023 (no 1-minute files listed)", "IWM": "no product", "SPX": "not listed on the pages checked"},
       pricing={"model": "per year, one-time", "summary": "Per symbol-year, read from the storefront: end of day $0, 30-minute $3, 15-minute $5, 5-minute $20, 1-minute $50. No refunds.",
                "tiers": [{"name": "End of day", "price": 0, "history": "2010-2023", "what": "one snapshot a day"}, {"name": "30 minutes", "price": 3, "history": "2010-2023", "what": "per year: bid/ask every 30 minutes"},
                          {"name": "15 minutes", "price": 5, "history": "2011-2023", "what": "per year"}, {"name": "5 minutes", "price": 20, "history": "2016-2023", "what": "per year"}, {"name": "1 minute", "price": 50, "history": "2020-2023", "what": "per year"}],
                "source_url": "https://www.optionsdx.com/product/spy-option-chains/"},
       limits=["Snapshots, not every quote: between two snapshots nothing is known, so an exit is only seen at the next one.", "Manual purchase and download; Strata never buys anything.",
               "The vendor does not say which feed the prices come from. For personal research; do not redistribute the files.", "Nothing after 2023, and no IWM."],
       integration="import", import_format="optionsdx",
       tested="verified on real data: the vendor's public sample file (SPY, 2020-03-06, 5-minute snapshots 09:30-12:00 ET, 38 expirations including the 0DTE) was imported and cross-checked. "
              "All 9,716 contracts appear in the independent end-of-day archive for that date, no quote is crossed, and the underlying matches the stored daily bar. A purchased full-session file was not tested"),
    _e("algoseek_aws", "Algoseek free samples on AWS Data Exchange", "sample files", source=None, url="https://aws.amazon.com/marketplace/pp/prodview-sk2vtzeo5noje", account="AWS account (free subscription to the listing)", credentials=[],
       data="US equity options trade+quote 1-minute bars built from the full OPRA feed (bid/ask and trades per minute)", resolution="1 minute", history="one month of 2020 (sample)",
       symbols={"SPY": "sample month only", "QQQ": "sample month only", "IWM": "sample month only", "SPX": "index options are a separate listing"},
       pricing={"model": "free sample", "summary": "The listing is offered free of charge; the full history is a commercial product.", "source_url": "https://aws.amazon.com/marketplace/pp/prodview-sk2vtzeo5noje"},
       limits=["A single month: enough to test an intraday bid/ask workflow, not to backtest.", "Requires an AWS account; Strata does not download it for you."], integration="import", import_format="generic option quotes (column mapping)",
       tested="not tested: no file from this listing was obtained"),
    _e("wrds_optionmetrics", "OptionMetrics IvyDB via WRDS", "academic", source=None, url="https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/", account="university subscription", credentials=[],
       data="End-of-day best bid/offer, volume, open interest, implied volatility and greeks for all US listed equity and index options", resolution="end of day", history="January 1996 onward (the academic copy usually lags by about a year)",
       symbols={"SPY": "1996 onward", "QQQ": "1999 onward", "IWM": "2000 onward", "SPX": "1996 onward"}, pricing={"model": "institutional", "summary": "No individual price: access comes with a university's WRDS subscription, for research and teaching only."},
       limits=["Available only to students and staff of subscribing institutions; non-commercial use.", "End of day only: it cannot price an intraday entry or a 0DTE trade.",
               "University of Washington: OptionMetrics is NOT in the list of WRDS datasets the Foster School publishes (checked 2026-11-25), so UW access to WRDS does not include it."], integration="import", import_format="wrds_optionmetrics",
       tested="import layout tested on a synthetic file with the documented WRDS column names; not tested on a real export"),
    _e("uw_wrds", "WRDS through the UW Foster School", "academic (University of Washington)", source=None, url="https://guides.lib.uw.edu/research/business/datasets", account="UW affiliation; accounts are managed by the Foster School, not the library", credentials=[],
       data="The Foster School's WRDS subscription lists TAQ (intraday trades and quotes for NYSE, AMEX and NASDAQ securities), CRSP, Compustat, IBES and others. No option-price dataset is listed: Cboe appears for VIX indexes only",
       resolution="tick (TAQ, stocks only)", history="per dataset", symbols={"SPY": "underlying only (TAQ)", "QQQ": "underlying only (TAQ)", "IWM": "underlying only (TAQ)", "SPX": "no option data"},
       pricing={"model": "institutional", "summary": "No individual price. Faculty and PhD students ask the Foster data research consultant for an account; undergraduate and master's students are told to ask their instructor.",
                "source_url": "https://ask.lib.uw.edu/business/faq/439846"},
       limits=["Could supply SPY/QQQ/IWM minute bars (from TAQ), not option prices.", "Licensed for academic use. UW's guidelines allow saving data for a single assignment or project and do not allow systematic downloading, automated searching or use with AI tools: "
               "export a file yourself for one project and import it; Strata must not log in or scrape.", "Whether an undergraduate gets an account is the instructor's and the school's decision."],
       integration="import", import_format="generic bars (column mapping)", tested="not tested: no account. Listed from UW's published pages (dataset list updated 2026-11-23)"),
    _e("uw_bloomberg", "Bloomberg terminals at UW", "academic (University of Washington)", source=None, url="https://ask.lib.uw.edu/business/faq/432118", account="UW affiliation; create a Bloomberg login at a terminal", credentials=[],
       data="Historical prices and option data on screen and through the Excel add-in, on the terminal only", resolution="depends on the function; intraday history on a terminal is limited to recent months",
       history="per function", symbols={"SPY": "on the terminal", "QQQ": "on the terminal", "IWM": "on the terminal", "SPX": "on the terminal"},
       pricing={"model": "institutional", "summary": "Free to use in the Foster Business Library (also Bothell and Tacoma), two-hour seat reservations. Not available off campus."},
       limits=["Terminal use only: there is no API or bulk download for students, and Bloomberg limits how much can be exported.", "Not a route to a multi-year intraday option history."],
       integration="none", tested="no integration (terminal only)"),
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
    "databento": {"per_minute": None, "batch": None, "price": None, "window": lambda today: (date(2013, 4, 1), today - timedelta(days=1)), "what": "1-minute bid/ask (CBBO-1m)", "record_bytes": 80, "per_gb": 2.0, "credit": 125.0},
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
    # did a contract the legs could use exist on the signal session? Only the stored end-of-day chains can establish that; where none is stored it is unknown
    absent, known, intraday_q = [], 0, 0
    try:
        eod = next((k for k in store.datasets(strat.symbol) if k.dtype == "options" and k.obs == "quote" and k.res == "EOD"), None)
        if eod is not None:
            exps = PL.expiration_index(store, eod)
            leg_specs = [l.model_dump() for l in legs]
            for d in days:
                if d in exps:
                    known += 1
                    if not all(PL._dte_ok(exps[d], l) for l in leg_specs):
                        absent.append(d)
        for k in store.datasets(strat.symbol):
            if k.dtype == "options" and k.obs == "quote" and k.res != "EOD":
                iq = {str(x) for x in store.option_days(strat.symbol, "quote", dataset=k)}
                intraday_q = max(intraday_q, sum(1 for d in days if d in iq))
    except Exception:  # noqa: BLE001
        pass
    existence = {"known_sessions": known, "unknown_sessions": len(days) - known, "no_contract": len(absent),
                 "note": (f"A contract with the required expiration is known to have existed (or not) on {known} of the {len(days)} signal sessions, from the stored end-of-day chains. "
                          + (f"On {len(absent)} of them no such expiration was listed: those sessions cannot be tested with any data. " if absent else "")
                          + (f"For the other {len(days) - known} it is not established here." if len(days) > known else ""))}
    return {"no_contract_days": absent, "contract_existence": existence, "sessions_with_intraday_quotes": intraday_q,
            "symbol": strat.symbol, "signals": len(signals), "signal_sessions": len(days), "first": days[0] if days else None, "last": days[-1] if days else None, "hold_sessions": hold, "option_legs": len(legs),
            "contracts_per_signal": contracts, "sessions_per_signal": hold + 1, "days": days, "pricing": mode, "sessions_with_quotes": sum(1 for d in days if d in have_q), "sessions_with_trades": sum(1 for d in days if d in have_t),
            "excluded_sessions": len({d for v in (sigs.get("excluded") or {}).values() for d in v}), "sessions_scanned": sigs.get("sessions"), # an end-of-day chain is one snapshot at the close: any entry before it (the open included) needs intraday option prices
            "intraday": any(int(s.get("minute", 570)) < 955 for s in signals) or any(l.dte == 0 for l in legs)}


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
            usd = gb * c["per_gb"]
            row.update(records=recs, gigabytes=round(gb, 4), dollars=round(usd, 2), dollars_floor=round(usd, 2), contract_sessions=pairs, credit=c["credit"], within_credit=usd <= c["credit"],
                       dollars_note=f"Estimate: {recs:,} one-minute records of about {c['record_bytes']} bytes at the documented ${c['per_gb']:g} per GB. " + ("It fits inside the $125 credit a new account receives. " if usd <= c["credit"] else "")
                       + "The provider returns the exact price, free of charge, before any download; nothing is requested without your approval of that figure.")
        elif sid == "thetadata":
            row.update(dollars=c["price"], dollars_note="one month of the Value tier (1-minute quotes from 2020); earlier dates need Standard ($80) or Pro ($160)", contract_sessions=pairs)
            if inside and min(inside) < date(2020, 1, 1):
                row["notes"].append("Some signal sessions are before 2020, which the Value tier does not reach.")
        if not inside and days:
            row["notes"].append("None of the signal sessions fall inside this source's history.")
        rows.append(row)
    rows.sort(key=lambda r: (-(r["coverage_pct"] or 0), r["dollars"] if r["dollars"] is not None else 1e9))
    return {"need": {k: v for k, v in need.items() if k not in ("days", "no_contract_days")}, "sources": rows,
            "notes": ["Signals are found by scanning the underlying only, so no option data is needed to produce this estimate.",
                      "Nothing here is started automatically. A free source is downloaded when you run the backtest (large downloads ask first); a paid one is never called without your approval of its quoted cost.",
                      "Dollar figures are the providers' published subscription prices on " + CHECKED + ", not charges Strata can see or control."]}


# ----------------------------------------------------------------------------- what a budget buys
BUDGETS = (0, 25, 50, 100, 200)


def _alpaca_first() -> date:
    from strata.data.providers.sources import ALPACA_OPTIONS_FIRST
    return ALPACA_OPTIONS_FIRST


def measured(provider: str = "alpaca") -> dict | None:
    """What the access test established with the key stored on this installation, or None when it has not returned an observation."""
    try:
        from strata.data.providers import access as A
        rep = A.saved(provider)
    except Exception:  # noqa: BLE001
        return None
    if not rep or not rep.get("verified"):
        return None
    cells = [c for row in rep.get("matrix") or [] for c in row.get("contracts") or []]
    dates = sorted({row["date"] for row in rep.get("matrix") or [] if any(c.get("has_observations") for c in row.get("contracts") or [])})
    return {"run_at": rep.get("run_at"), "contract_sessions": len(cells), "with_observations": sum(1 for c in cells if c.get("has_observations")), "first": dates[0] if dates else None, "last": dates[-1] if dates else None,
            "earliest": (rep.get("findings") or {}).get("earliest_option_bars")}


def _o(id_, entry, name, price, unit, obs, minutes, **kw):
    return {"id": id_, "entry": entry, "name": name, "price": price, "unit": unit, "observation": obs, "resolution_minutes": minutes, **kw}


# Every way found to obtain intraday option prices, as something that can be bought (or had free). Prices and coverage were read from each
# provider's own pages on CHECKED. `window` is the history it reaches; `years` are the calendar years a per-year product is sold for.
OFFERS = [
    _o("alpaca_free", "alpaca", "Alpaca free account", 0.0, "free account", "trade prints", 1, window=lambda t: (_alpaca_first(), t), symbols=None, route="built-in: connect the key in Settings",
       needs="a free Alpaca account (paper is enough)", caveat="Trade prints only: no bid/ask. Never run with a real key on this installation."),
    _o("massive_basic", "massive", "Massive (Polygon) free plan", 0.0, "free account", "trade prints", 1, window=lambda t: (t - timedelta(days=730), t - timedelta(days=1)), symbols=None, route="built-in adapter: add the key in Settings",
       needs="a free Massive account", caveat="Trade prints only; 5 requests a minute. Never run with a real key on this installation."),
    _o("databento_credit", "databento", "Databento free credit", 0.0, "usage inside a $125 credit", "bid/ask quotes", 1, window=lambda t: (date(2013, 4, 1), t - timedelta(days=1)), symbols=None, usage=True,
       route="quote first, then download only the contracts the strategy needs", needs="a Databento account (new accounts receive $125 of credit)",
       caveat="The credit is spent by the byte and expires after 6 months. The exact price is quoted by the provider before anything is downloaded. Never run with a real key on this installation."),
    _o("optionsdx_30m", "optionsdx", "OptionsDX 30-minute chains", 3.0, "per symbol-year", "bid/ask quotes", 30, years=(2010, 2023), symbols={"SPY": (2010, 2023), "QQQ": (2012, 2023)}, route="buy and download the files, then import them (Data, Import)",
       needs="an OptionsDX account and a manual purchase", caveat="A snapshot every 30 minutes: an exit is only seen at the next snapshot."),
    _o("optionsdx_15m", "optionsdx", "OptionsDX 15-minute chains", 5.0, "per symbol-year", "bid/ask quotes", 15, years=(2011, 2023), symbols={"SPY": (2011, 2023), "QQQ": (2012, 2023)}, route="buy and download the files, then import them (Data, Import)",
       needs="an OptionsDX account and a manual purchase", caveat="A snapshot every 15 minutes."),
    _o("optionsdx_5m", "optionsdx", "OptionsDX 5-minute chains", 20.0, "per symbol-year", "bid/ask quotes", 5, years=(2016, 2023), symbols={"SPY": (2016, 2023), "QQQ": (2016, 2023)}, route="buy and download the files, then import them (Data, Import)",
       needs="an OptionsDX account and a manual purchase", caveat="A snapshot every 5 minutes. The importer was verified on the vendor's public 5-minute sample."),
    _o("optionsdx_1m", "optionsdx", "OptionsDX 1-minute chains", 50.0, "per symbol-year", "bid/ask quotes", 1, years=(2020, 2023), symbols={"SPY": (2020, 2023)}, route="buy and download the files, then import them (Data, Import)",
       needs="an OptionsDX account and a manual purchase", caveat="About 20 GB of CSV per year."),
    _o("thetadata_value", "thetadata", "ThetaData Value, one month", 40.0, "per month", "bid/ask quotes", 1, window=lambda t: (date(2020, 1, 1), t), symbols=None, route="adapter exists; never run against the live service",
       needs="a ThetaData subscription and their terminal program", caveat="A subscription month: download what you need while it is active."),
    _o("thetadata_standard", "thetadata", "ThetaData Standard, one month", 80.0, "per month", "bid/ask quotes", 1, window=lambda t: (date(2016, 1, 1), t), symbols=None, route="adapter exists; never run against the live service",
       needs="a ThetaData subscription and their terminal program", caveat="Adds trades and tick data; history from 2016."),
    _o("thetadata_pro", "thetadata", "ThetaData Pro, one month", 160.0, "per month", "bid/ask quotes", 1, window=lambda t: (date(2012, 6, 1), t), symbols=None, route="adapter exists; never run against the live service",
       needs="a ThetaData subscription and their terminal program", caveat="History from June 2012."),
    _o("massive_developer", "massive", "Massive Developer, one month", 79.0, "per month", "trade prints", 1, window=lambda t: (t - timedelta(days=1461), t), symbols=None, route="built-in adapter: add the key in Settings",
       needs="a Massive subscription", caveat="Trades and trade bars for 4 years; still no bid/ask."),
    _o("massive_advanced", "massive", "Massive Advanced, one month", 199.0, "per month", "bid/ask quotes", 1, window=lambda t: (date(2022, 3, 7), t), symbols=None, route="adapter reads trade bars only; quotes are not integrated",
       needs="a Massive subscription", caveat="Quote history starts 2022-03-07 (trade bars from 2014)."),
]
NOT_OFFERS = ["Alpaca Algo Trader Plus ($99 a month) adds the live OPRA feed, not history: Alpaca has no historical option quotes on any plan and its option history still starts in January 2024.",
              "End-of-day archives (the free ones Strata already downloads, OptionMetrics through a university, Alpha Vantage, EODHD, MarketData.app) cannot price an intraday entry or a 0DTE trade, whatever they cost.",
              "Cboe DataShop and ORATS sell complete interval histories; neither publishes a price that fits these budgets (ORATS lists $1,500 for its 1-minute history)."]


def _window(o: dict, sym: str, today: date) -> tuple[date, date] | None:
    if o.get("symbols") is not None:
        y = o["symbols"].get(sym)
        return (date(y[0], 1, 1), date(y[1], 12, 31)) if y else None
    return o["window"](today)


def budget_table(need: dict, today: date | None = None, budgets=BUDGETS) -> dict:
    """What each budget would make testable, for the sessions THIS strategy needs.

    `need` comes from `needs_of`: the sessions on which the entry rule fires, found from the underlying alone. For
    every offer the sessions inside its history are counted; for every budget the purchase that covers the most of
    them is listed, bid/ask before trade prints. A session a contract did not exist on stays untestable whatever is
    bought, and is kept apart where the stored end-of-day chains establish it."""
    today = today or date.today()
    sym = need.get("symbol") or "SPY"
    days = [date.fromisoformat(d) for d in need.get("days", [])]
    absent = {date.fromisoformat(d) for d in need.get("no_contract_days", [])}
    testable = [d for d in days if d not in absent]
    pairs_per = max(1, int(need.get("contracts_per_signal") or 1)) * max(1, int(need.get("sessions_per_signal") or 1))
    rate = COST["databento"]
    offers = []
    got_alpaca = measured("alpaca")
    for o in OFFERS:
        w = _window(o, sym, today)
        inside = [d for d in testable if w and w[0] <= d <= w[1]]
        row = {k: v for k, v in o.items() if k not in ("window", "symbols", "years")}
        row.update(window=[w[0].isoformat(), w[1].isoformat()] if w else None, sessions=len(inside), of=len(testable), pct=round(len(inside) / len(testable) * 100, 1) if testable else None, _days=inside)
        if o.get("years"):
            by_year: dict[int, int] = {}
            for d in inside:
                by_year[d.year] = by_year.get(d.year, 0) + 1
            row["by_year"] = dict(sorted(by_year.items()))
        if o.get("usage"):
            gb = len(inside) * pairs_per * 390 * rate["record_bytes"] / 1e9
            row.update(estimated_cost=round(gb * rate["per_gb"], 2), estimated_gb=round(gb, 3), credit=rate["credit"], contract_sessions=len(inside) * pairs_per,
                       cost_note=f"about ${gb * rate['per_gb']:,.2f} of the ${rate['credit']:g} credit for {len(inside) * pairs_per:,} contract-sessions (estimate; the provider quotes the exact figure)")
        if not w:
            row["caveat"] = f"No {sym} product. " + row.get("caveat", "")
        if o["entry"] == "alpaca" and o["price"] == 0 and got_alpaca:
            row.update(verified=True, route="built-in: the key is connected, and a run downloads the contracts it needs by itself",
                       caveat=f"Trade prints only: no bid/ask. Measured with the key stored here on {str(got_alpaca['run_at'])[:10]}: {got_alpaca['with_observations']} of {got_alpaca['contract_sessions']} expired contract-sessions asked for "
                              f"returned 1-minute trade bars ({got_alpaca['first']} to {got_alpaca['last']}); the first session with any is {_alpaca_first()}.")
        offers.append(row)
    rows = []
    for b in budgets:
        plans = []
        free = [o for o in offers if o["price"] == 0 and o["sessions"]]
        free_q = {d for o in free if o["observation"] == "bid/ask quotes" for d in o["_days"]}
        free_t = {d for o in free for d in o["_days"]}
        if free:
            plans.append({"label": " + ".join(o["name"] for o in free), "cost": 0.0, "items": [{"offer": o["id"], "name": o["name"], "cost": 0.0, "sessions": o["sessions"], "note": o.get("cost_note")} for o in free],
                          "quote_days": free_q, "any_days": free_t, "minutes": min(o["resolution_minutes"] for o in free), "needs": sorted({o["needs"] for o in free})})
        for o in offers:
            if o["price"] <= 0 or o["price"] > b or not o["sessions"]:
                continue
            if "by_year" in o:  # buy the years with the most signal sessions first
                picked, cost = [], 0.0
                for y, n in sorted(o["by_year"].items(), key=lambda kv: (-kv[1], -kv[0])):
                    if cost + o["price"] <= b + 1e-9:
                        picked.append(y)
                        cost += o["price"]
                got = {d for d in o["_days"] if d.year in set(picked)}
                if got:
                    plans.append({"label": f"{o['name']}: {_years(sorted(picked))}", "cost": cost, "items": [{"offer": o["id"], "name": o["name"], "years": sorted(picked), "cost": cost, "sessions": len(got)}],
                                  "quote_days": got, "any_days": got, "minutes": o["resolution_minutes"], "needs": [o["needs"]]})
            else:
                got = set(o["_days"])
                plans.append({"label": o["name"], "cost": o["price"], "items": [{"offer": o["id"], "name": o["name"], "cost": o["price"], "sessions": len(got)}],
                              "quote_days": got if o["observation"] == "bid/ask quotes" else set(), "any_days": got, "minutes": o["resolution_minutes"], "needs": [o["needs"]]})
        out = []
        for pl_ in plans:
            q_all = pl_["quote_days"] | (free_q if pl_["cost"] else set())
            a_all = pl_["any_days"] | (free_t if pl_["cost"] else set())
            out.append({"label": pl_["label"] + (" (plus the free sources)" if pl_["cost"] and free else ""), "cost": round(pl_["cost"], 2), "items": pl_["items"], "needs": pl_["needs"], "resolution_minutes": pl_["minutes"],
                        "sessions_with_bid_ask": len(q_all), "sessions_trade_prints_only": len(a_all - q_all), "sessions": len(a_all), "of": len(testable), "pct": round(len(a_all) / len(testable) * 100, 1) if testable else None,
                        "first": min(a_all).isoformat() if a_all else None, "last": max(a_all).isoformat() if a_all else None})
        out.sort(key=lambda r: (-r["sessions_with_bid_ask"], -r["sessions"], r["resolution_minutes"], r["cost"]))
        # the same budget at each granularity: a 30-minute snapshot reaches further back, a 1-minute one sees an intraday exit when it happens
        by_res = []
        for cap, label in ((1, "1-minute"), (5, "5-minute or finer"), (30, "30-minute or finer")):
            fit = [r for r in out if r["resolution_minutes"] <= cap]
            if fit:
                by_res.append({"granularity": label, "max_minutes": cap, **{k: fit[0][k] for k in ("label", "cost", "sessions_with_bid_ask", "sessions_trade_prints_only", "sessions", "of", "pct", "first", "last", "needs", "items")}})
        rows.append({"budget": b, "best": out[0] if out else None, "by_granularity": by_res, "alternatives": out[1:4], "nothing": not out})
    # the earliest day ANY compared source reaches: sessions before it are outside all of them, and saying "N of M" without
    # saying so would make every source look worse than it is while hiding that the early years cannot be bought at all
    starts = [date.fromisoformat(o["window"][0]) for o in offers if o.get("window")]
    earliest = min(starts) if starts else None
    before = [d for d in testable if earliest and d < earliest]
    for o in offers:
        o.pop("_days", None)
    have_q, have_t = int(need.get("sessions_with_quotes") or 0), int(need.get("sessions_with_trades") or 0)
    known = (need.get("contract_existence") or {}).get("known_sessions")
    return {"checked": CHECKED, "symbol": sym, "signal_sessions": len(days), "testable_sessions": len(testable), "no_contract_sessions": len(absent), "contract_existence": need.get("contract_existence"),
            "earliest_source_date": earliest.isoformat() if earliest else None, "before_any_source": len(before), "reachable_sessions": len(testable) - len(before),
            "denominator_note": (f"{len(before):,} of the {len(testable):,} sessions fall before {earliest.isoformat()}, the earliest day any source compared here reaches: no budget buys them. " if before else "")
            + (f"Whether a contract with the required expiration was listed is established for {known:,} of the {len(days):,} sessions only (from stored end-of-day chains), so every count is an upper bound." if known is not None else
               "Whether a contract with the required expiration was listed on each session is not established here, so every count is an upper bound."),
            "stored_now": {"sessions_with_intraday_quotes": int(need.get("sessions_with_intraday_quotes") or 0), "sessions_with_trade_prints": have_t, "sessions_with_end_of_day_quotes": have_q},
            "contract_sessions_per_signal": pairs_per, "budgets": rows, "offers": offers, "not_offers": NOT_OFFERS,
            "notes": ["Counts are this strategy's qualifying sessions inside each source's history, not a promise that every contract traded or was quoted on them.",
                      (f"Alpaca's coverage was measured with the key stored here ({str(got_alpaca['run_at'])[:10]}). The other paid or key-gated routes have not been run from this installation: their coverage is what the provider publishes. "
                       if got_alpaca else "None of the paid or key-gated routes has been run from this installation: their coverage is what the provider publishes. ")
                      + "The OptionsDX importer was verified on the vendor's own sample file.",
                      "Strata never buys anything and never starts a paid request. A purchase is yours to make; a metered download asks for approval of the provider's own quote first.",
                      f"Prices were read from the providers' pages on {CHECKED}."]}


def _years(ys: list[int]) -> str:
    out, i = [], 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and ys[j + 1] == ys[j] + 1:
            j += 1
        out.append(str(ys[i]) if i == j else f"{ys[i]}-{ys[j]}")
        i = j + 1
    return ", ".join(out)


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
