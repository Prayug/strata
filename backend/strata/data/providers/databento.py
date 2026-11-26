"""Databento OPRA: historical bid/ask for option contracts, connected in stages so that nothing is spent by accident.

STATUS (2026-11-26): no Databento key has ever been stored in this installation. Everything below was written against the
provider's published HTTP API and its documented prices, and has NOT been run against the live service. The parsers are
exercised only through a mocked transport. Until `strata access databento` has run with a real key this module establishes
nothing about what an account can retrieve, and every report it produces without one says BLOCKED.

The stages, each of which stops and reports before the next:

  0  key        is a key stored? (never printed, never sent anywhere but the provider)
  1  metadata   free of charge: is OPRA.PILLAR visible to this key, what period does each schema cover, what does a
                gigabyte cost (metadata.list_datasets, metadata.get_dataset_range, metadata.list_unit_prices)
  2  quote      free of charge: the exact price of the representative sample, from the provider itself (metadata.get_cost)
  3  sample     BILLABLE: only when the caller passes a budget that covers the quoted price. One-minute consolidated
                bid/ask (cbbo-1m) for the contracts the sample sessions need, stored as a quote dataset that says it is
                a band of strikes, not a chain
  4  alignment  the downloaded quotes against the stored Alpaca trade bars of the same contracts and minutes
  5  execution  the same strategy on trade prints and on the quotes

Stages 4 and 5 are `strata.research.obscompare` and work on any stored pair of datasets (they are tested on fixtures).
"""
from __future__ import annotations

import io
import json
from datetime import date, datetime, timedelta, timezone

import polars as pl

from strata.core import calendar as cal
from strata.data.providers import sources as S
from strata.data.store import MarketStore

BASE = "https://hist.databento.com/v0"
DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1m"
CREDIT = 125.0
# From the provider's pages, read on 2026-11-26. They are instructions for a person: Strata never creates an account or
# types a key.
SETUP = [
    "Open https://databento.com and choose Sign up to create an account.",
    "Databento states that new users receive $125 of free credit, usable for historical data, shared by the whole team, applied before any charge and expiring six months after it is issued.",
    "Sign in and open the API keys page of the portal (https://databento.com/portal/keys). A key is created with the account: a 32-character string that starts with db-.",
    "Before the first download open the Billing page of the portal (https://databento.com/portal/billing). Databento's documentation says the remaining credit is shown there and that the account has a monthly usage limit you can set; "
    "it says further historical requests are blocked once the month's usage exceeds that limit. Set the limit to an amount you accept. Strata cannot see the credit or the limit: neither has an API.",
    "In Strata open Settings, \"Data provider keys\", paste the key into the Databento field and press Save. The key goes to the operating system's credential store and is never shown again.",
    "Run the check: Data -> Readiness -> \"Check Databento\" or `strata access databento` in a terminal. It makes free metadata requests only and ends with the exact price of a small sample, quoted by Databento itself.",
    "To download that sample, run it again with a budget that covers the quoted price, for example `strata access databento --budget 1`. Nothing billable is requested without a budget, and never more than it.",
]
PRICES_DOCUMENTED = {"cbbo-1m": 2.0, "cbbo-1s": 2.0, "tcbbo": 210.0, "trades": 280.0, "ohlcv-1m": 280.0, "definition": 5.0}  # USD per GB, historical, from metadata.list_unit_prices' documented example
NOTES = ["Databento's documentation says OPRA data before 2023-03-28 was not generated from packet captures but from subsampled data; the cbbo-1s, cmbp-1 and tcbbo schemas do not exist for that period. cbbo-1m is listed for the whole history.",
         "A cbbo-1m record is the consolidated best bid and offer sampled once a minute. It is a quote, not a trade: a fill priced from it assumes the quoted size was there for this order.",
         "The remaining credit is shown only in the provider's portal: there is no API endpoint for it, so Strata tracks what its own requests were quoted and stops at the budget it was given."]


def _key() -> str:
    from strata.api.settings import get_secret
    return get_secret("DATABENTO_API_KEY") or ""


def osi(sym: str, exp: date, right: str, strike: float) -> str:
    """The 21-character OSI symbol OPRA uses as a raw symbol: the root padded to six characters, then yymmdd, C or P, and the
    strike in thousandths. 'SPY   240119C00470000'."""
    return f"{sym:<6}{exp:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def from_osi(raw: str) -> tuple[str, date, str, float]:
    s = raw.strip()
    root, tail = raw[:6].strip(), raw[6:].strip()
    if len(tail) != 15:  # tolerate a single-space form
        root, tail = s[:-15].strip(), s[-15:]
    return root, datetime.strptime(tail[:6], "%y%m%d").date(), tail[6], int(tail[7:]) / 1000.0


def _ts(text: str) -> datetime:
    """'2024-01-19T14:31:00.000000000Z' (pretty_ts) -> an aware datetime, to the microsecond."""
    t = text.strip().rstrip("Z")
    if "+" in t[10:]:
        t = t[:t.index("+", 10)]
    head, _, frac = t.partition(".")
    return datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc, microsecond=int((frac + "000000")[:6]))


def _get(client, path: str, params: dict) -> object:
    """One GET against the historical API. The key is the basic-auth user name with an empty password, as documented."""
    r = client.get(f"{BASE}/{path}", params=params, auth=(_key(), ""), timeout=60)
    if r.status_code != 200:
        body = (r.text or "")[:300].replace(_key(), "[key]") if _key() else (r.text or "")[:300]
        raise S.SourceError(f"Databento {path}: HTTP {r.status_code} {body}")
    return r


def sample_plan(store: MarketStore, symbol: str = "SPY", sessions: int = 20, trade_provider: str = "alpaca") -> dict:
    """A small representative sample: the sessions of the original strategy that the stored trade data prices, plus sessions
    spread evenly over the stored period, and for each the at-the-money same-day calls around the open (the contracts the
    strategy trades) with their neighbours. Chosen from what is stored, so that every quote can be set beside a trade bar."""
    idx = store.option_bar_index(symbol, provider=trade_provider)
    if idx.is_empty():
        return {"sessions": [], "symbols": 0, "why": f"no option trade bars are stored for {symbol} from {trade_provider}: there would be nothing to compare the quotes with"}
    part = {date.fromisoformat(k) for k, v in store.session_states(store.option_dataset(symbol, "trade", trade_provider)).items() if v.get("status") == "partial"}
    days = sorted(set(idx["date"].unique().to_list()) - part)
    daily = store.read_daily(symbol).select("date", "open", "close").sort("date").with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("ret").shift(1).alias("prev_ret"))
    down = [d for d in daily.filter(pl.col("prev_ret") <= -0.03)["date"].to_list() if d in set(days)]
    step = max(1, len(days) // max(1, sessions - len(down)))
    spread = [d for d in days[::step] if d not in down][: max(0, sessions - len(down))]
    chosen = sorted(set(down) | set(spread))[:sessions]
    opens = {r["date"]: r["open"] for r in daily.filter(pl.col("date").is_in(chosen)).to_dicts()}
    rows, total = [], 0
    for d in chosen:
        day = idx.filter((pl.col("date") == d) & (pl.col("expiration") == d) & (pl.col("right") == "C") & (pl.col("status") == "ok"))
        if day.is_empty() or opens.get(d) is None:
            continue
        ks = sorted(day["strike"].to_list(), key=lambda k: abs(k - opens[d]))[:7]
        rows.append({"date": d.isoformat(), "why": "a 3% down day came before it" if d in down else "spread over the stored period", "open": round(float(opens[d]), 2),
                     "contracts": [osi(symbol, d, "C", k) for k in sorted(ks)]})
        total += len(ks)
    return {"symbol": symbol, "sessions": rows, "symbols": total, "schema": SCHEMA, "dataset": DATASET, "records_at_most": total * 390,
            "what": f"one-minute bid/ask for {total} same-day call contracts on {len(rows)} sessions between {rows[0]['date'] if rows else '-'} and {rows[-1]['date'] if rows else '-'}"}


def parse_cbbo_csv(text: str, d: date) -> pl.DataFrame:
    """The provider's CSV for cbbo-1m (requested with pretty_px, pretty_ts and map_symbols) -> Strata's quote snapshots.
    The header is read, not assumed: a column that is not there is an error naming the header that came back."""
    if not text.strip():
        return pl.DataFrame()
    try:
        df = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    except Exception as e:  # noqa: BLE001  (an answer that is not the CSV asked for)
        raise S.SourceError(f"Databento's {SCHEMA} answer could not be read as CSV ({type(e).__name__}); it begins: {text[:160]!r}. Nothing was stored.") from None
    need = ["symbol", "bid_px_00", "ask_px_00"]
    tcol = next((c for c in ("ts_recv", "ts_event") if c in df.columns), None)
    miss = [c for c in need if c not in df.columns]
    if miss or tcol is None:
        raise S.SourceError(f"Databento's {SCHEMA} CSV does not have the columns this adapter was written for (missing: {', '.join(miss + ([] if tcol else ['ts_recv or ts_event']))}; the header was: {', '.join(df.columns)[:300]}). Nothing was stored.")
    rows = []
    for r in df.iter_rows(named=True):
        try:
            _root, exp, right, strike = from_osi(r["symbol"])
            bid, ask = float(r["bid_px_00"] or "nan"), float(r["ask_px_00"] or "nan")
        except (ValueError, TypeError):
            continue
        try:
            ts = _ts(str(r[tcol]))
        except ValueError:
            continue
        rows.append({"ts": ts.replace(second=0, microsecond=0), "quote_ts": ts, "expiration": exp, "strike": strike, "right": right, "bid": bid, "ask": ask,
                     "bid_size": int(float(r.get("bid_sz_00") or 0)), "ask_size": int(float(r.get("ask_sz_00") or 0))})
    if not rows:
        return pl.DataFrame()
    out = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")), pl.col("quote_ts").cast(pl.Datetime("us", "UTC")), pl.col("bid_size").cast(pl.Int32), pl.col("ask_size").cast(pl.Int32))
    a, b = (x.astimezone(timezone.utc) for x in cal.session_bounds(d))
    return out.filter((pl.col("ts") >= a) & (pl.col("ts") <= b) & pl.col("bid").is_finite() & pl.col("ask").is_finite() & (pl.col("ask") >= pl.col("bid")) & (pl.col("ask") > 0))


def access(store: MarketStore | None = None, symbol: str = "SPY", sessions: int = 20, budget_usd: float = 0.0, progress=None, save: bool = True) -> dict:
    """Run the stages as far as the key and the budget allow. Returns the report; never raises for a provider refusal."""
    store = store or MarketStore()
    rep: dict = {"provider": "databento", "symbol": symbol, "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "blocked", "verified": False, "stages": [], "budget_usd": float(budget_usd),
                 "spent_usd": 0.0, "notes": NOTES, "unverified": "This adapter has never run against the live service: until it has, what it would retrieve is the provider's documentation, not a measurement."}
    key = _key()
    if not key:
        rep.update(reason="BLOCKED: no Databento key is stored in Strata's credential store, the environment or a .env file.", needs=["DATABENTO_API_KEY"], how=SETUP,
                   documented={"credit_usd": CREDIT, "unit_prices_usd_per_gb": PRICES_DOCUMENTED, "dataset": DATASET, "schema": SCHEMA})
        plan = sample_plan(store, symbol, sessions)
        rep["sample"] = {**plan, "sessions": plan["sessions"][:40], "estimate_usd": round(plan.get("records_at_most", 0) * 80 / 1e9 * PRICES_DOCUMENTED["cbbo-1m"], 4),
                         "estimate_note": "an estimate from the documented price and about 80 bytes a record; with a key the provider quotes the exact price free of charge"}
        rep["stages"].append({"stage": "key", "ok": False, "detail": "no key stored"})
        return _finish(rep, save)
    rep["stages"].append({"stage": "key", "ok": True, "detail": "a key is stored" + ("" if key.startswith("db-") and len(key) == 32 else " (it does not look like a Databento key: those are 32 characters starting with db-)")})
    with S._client() as client:
        try:
            if progress:
                progress(0.1, "metadata (free)")
            ds = _get(client, "metadata.list_datasets", {}).json()
            ok = DATASET in ds
            rep["stages"].append({"stage": "datasets", "ok": ok, "detail": f"{len(ds)} datasets are visible to this key; {DATASET} is {'among them' if ok else 'NOT among them'}"})
            if not ok:
                rep.update(status="not_entitled", reason=f"The key works but {DATASET} is not in the datasets it can see.")
                return _finish(rep, save)
            rng = _get(client, "metadata.get_dataset_range", {"dataset": DATASET}).json()
            sch = (rng.get("schema") or {}).get(SCHEMA) or {}
            rep["range"] = {"start": rng.get("start"), "end": rng.get("end"), SCHEMA: sch}
            rep["stages"].append({"stage": "range", "ok": bool(sch), "detail": f"{SCHEMA}: {sch.get('start', '?')[:10]} to {sch.get('end', '?')[:10]}" if sch else f"the range answer has no entry for {SCHEMA}"})
            prices = _get(client, "metadata.list_unit_prices", {"dataset": DATASET}).json()
            per_gb = next((float(p["unit_prices"][SCHEMA]) for p in prices if p.get("mode") == "historical-streaming" and SCHEMA in (p.get("unit_prices") or {})), None)
            rep["unit_price_usd_per_gb"] = per_gb
            rep["stages"].append({"stage": "prices", "ok": per_gb is not None, "detail": f"{SCHEMA} costs ${per_gb:g} per GB (documented: ${PRICES_DOCUMENTED[SCHEMA]:g})" if per_gb is not None else "no unit price was returned for the schema"})
            plan = sample_plan(store, symbol, sessions)
            quoted = 0.0
            for i, srow in enumerate(plan["sessions"]):
                if progress:
                    progress(0.2 + 0.3 * i / max(1, len(plan["sessions"])), f"price of {srow['date']} (free)")
                d = date.fromisoformat(srow["date"])
                a, b = (x.astimezone(timezone.utc) for x in cal.session_bounds(d))
                c = float(_get(client, "metadata.get_cost", {"dataset": DATASET, "schema": SCHEMA, "stype_in": "raw_symbol", "symbols": ",".join(srow["contracts"]),
                                                                "start": a.strftime("%Y-%m-%dT%H:%M"), "end": (b + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M")}).json())
                srow["quoted_usd"] = c
                quoted += c
            rep["sample"] = {**plan, "quoted_usd": round(quoted, 4)}
            rep["stages"].append({"stage": "quote", "ok": True, "detail": f"the sample ({plan['what']}) is priced at ${quoted:,.4f} by the provider"})
            rep.update(status="quoted", verified=False)
            if budget_usd <= 0 or quoted > budget_usd + 1e-9:
                rep["reason"] = (f"Stopped before the first billable request: the sample costs ${quoted:,.4f} and " + ("no budget was given." if budget_usd <= 0 else f"the budget is ${budget_usd:,.2f}.")
                                 + f" Run it again with a budget of at least ${quoted:,.2f} to download it (the provider applies free credit first; the remaining credit is only shown in its portal).")
                return _finish(rep, save)
            got = 0
            for i, srow in enumerate(plan["sessions"]):
                if progress:
                    progress(0.5 + 0.5 * i / max(1, len(plan["sessions"])), f"downloading {srow['date']}")
                d = date.fromisoformat(srow["date"])
                a, b = (x.astimezone(timezone.utc) for x in cal.session_bounds(d))
                r = _get(client, "timeseries.get_range", {"dataset": DATASET, "schema": SCHEMA, "stype_in": "raw_symbol", "symbols": ",".join(srow["contracts"]), "start": a.strftime("%Y-%m-%dT%H:%M"),
                                                          "end": (b + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M"), "encoding": "csv", "pretty_px": "true", "pretty_ts": "true", "map_symbols": "true"})
                df = parse_cbbo_csv(r.text, d)
                rep["spent_usd"] = round(rep["spent_usd"] + float(srow["quoted_usd"]), 4)
                if df.is_empty():
                    srow["rows"] = 0
                    continue
                store.write_option_day(symbol, d, df, provider="databento", res="1m", status="complete", source=f"Databento {DATASET} {SCHEMA}", complete_chain=False,
                                       observation="consolidated best bid and offer sampled once a minute", limitations=["A band of contracts, not the chain: the at-the-money same-day calls around the open.", NOTES[1]])
                srow["rows"] = df.height
                got += df.height
            S._finish(store, symbol)
            rep["stages"].append({"stage": "sample", "ok": got > 0, "detail": f"{got:,} quote records stored for {sum(1 for x in plan['sessions'] if x.get('rows'))} sessions; ${rep['spent_usd']:,.4f} of the ${budget_usd:,.2f} budget used"})
            rep.update(status="sampled", verified=got > 0)
        except S.SourceError as e:
            rep["stages"].append({"stage": "request", "ok": False, "detail": str(e)[:300]})
            rep.update(status="failed", reason=str(e)[:300])
        except (ValueError, KeyError, TypeError) as e:
            rep["stages"].append({"stage": "answer", "ok": False, "detail": f"the provider's answer was not what this adapter was written for ({type(e).__name__}: {str(e)[:200]})"})
            rep.update(status="failed", reason="The provider answered in a shape this adapter does not know. It was written from the documentation and never run live: the first raw answer is the place to look.")
    if rep["status"] == "sampled":
        from strata.research import obscompare as OC
        rep["alignment"] = OC.alignment(store, symbol, "databento", "alpaca")
    return _finish(rep, save)


def _finish(rep: dict, save: bool) -> dict:
    rep["summary"] = render(rep).splitlines()
    if save:
        from strata.config import get_settings
        p = get_settings().data_dir / "access"
        p.mkdir(parents=True, exist_ok=True)
        (p / "databento.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def saved() -> dict | None:
    from strata.config import get_settings
    p = get_settings().data_dir / "access" / "databento.json"
    return json.loads(p.read_text()) if p.exists() else None


def render(rep: dict) -> str:
    out = [f"Databento ({DATASET}, {SCHEMA}): {rep['status'].upper().replace('_', ' ')}"]
    if rep.get("reason"):
        out.append(rep["reason"])
    for st in rep.get("stages") or []:
        out.append(f"  [{'ok' if st['ok'] else '--'}] {st['stage']}: {st['detail']}")
    sm = rep.get("sample") or {}
    if sm.get("what"):
        out.append(f"  sample: {sm['what']}" + (f"; quoted ${sm['quoted_usd']:,.4f} by the provider" if sm.get("quoted_usd") is not None else f"; about ${sm.get('estimate_usd', 0):,.4f} at the documented price (not a quote)" if sm.get("estimate_usd") is not None else ""))
    if rep.get("how"):
        out.append("To connect it:")
        out += [f"  {i + 1}. {s}" for i, s in enumerate(rep["how"])]
    al = rep.get("alignment") or {}
    if al.get("text"):
        out.append("  " + al["text"])
    out.append("  " + rep["unverified"] if not rep.get("verified") else "  The sample was downloaded with this key: the stages above are measurements.")
    return "\n".join(out)
