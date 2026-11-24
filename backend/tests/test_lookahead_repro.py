"""Look-ahead prevention, reproducibility and calendar handling on a synthetic window
(Feb-Apr 2024: includes the DST change on Mar 10 and the Good Friday holiday)."""
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from strata.core import calendar as cal
from strata.data.store import MarketStore
from strata.data.synthetic import SyntheticConfig, UnderlyingSpec, generate_demo_dataset
from strata.engine.engine import run_backtest


@pytest.fixture(scope="module")
def synth_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    st = MarketStore(root / "market")
    cfg = SyntheticConfig(start=date(2023, 11, 1), end=date(2024, 4, 30), seed=11,
                          underlyings=[UnderlyingSpec("DEMO", 400.0, "etf", max_dte=40, weekly_dte=35)])
    generate_demo_dataset(st, cfg, log=lambda *a: None)
    return st


class PerturbedStore(MarketStore):
    """Returns different data for everything observable after `cut` (epoch s)."""

    def __init__(self, base: MarketStore, cut: int):
        super().__init__(base.root)
        self.cut = cut

    def read_minutes(self, sym, start=None, end=None):
        df = super().read_minutes(sym, start, end)
        after = pl.col("ts").dt.epoch("s") + 60 > self.cut  # bars not yet complete at the cut
        return df.with_columns([pl.when(after).then(pl.col(c) * 1.07).otherwise(pl.col(c)).alias(c) for c in ["open", "high", "low", "close"]])

    def read_option_day(self, sym, d):
        df = super().read_option_day(sym, d)
        if df is None:
            return None
        after = pl.col("ts").dt.epoch("s") > self.cut
        return df.with_columns(
            pl.when(after).then(pl.col("bid") * 0.6).otherwise(pl.col("bid")).alias("bid"),
            pl.when(after).then(pl.col("ask") * 1.5 + 0.05).otherwise(pl.col("ask")).alias("ask"),
            pl.when(after).then(pl.col("delta") * 0.7).otherwise(pl.col("delta")).alias("delta"),
        )


STRATEGY = {
    "name": "lookahead probe", "symbol": "DEMO",
    "legs": [
        {"id": "sp", "side": "sell", "right": "P", "dte": 0, "strike": {"method": "delta", "value": 0.2}},
        {"id": "lp", "side": "buy", "right": "P", "dte": 0, "strike": {"method": "leg_offset", "value": -5, "ref_leg": "sp"}},
        {"id": "sc", "side": "sell", "right": "C", "dte": 0, "strike": {"method": "delta", "value": 0.2}},
        {"id": "lc", "side": "buy", "right": "C", "dte": 0, "strike": {"method": "leg_offset", "value": 5, "ref_leg": "sc"}},
    ],
    "entry": {
        "conditions": {"kind": "group", "op": "and", "children": [
            {"kind": "cond", "type": "entry_times", "params": {"times": "10:00,11:00,12:30,14:00"}},
            {"kind": "group", "op": "or", "children": [
                {"kind": "cond", "type": "compare", "params": {"lhs": {"kind": "indicator", "name": "ema", "params": {"period": 9}, "tf": "5m"}, "cmp": ">", "rhs": {"kind": "indicator", "name": "ema", "params": {"period": 21}, "tf": "5m"}}},
                {"kind": "cond", "type": "opening_range", "params": {"minutes": 30, "direction": "inside"}},
                {"kind": "cond", "type": "compare", "params": {"lhs": {"kind": "indicator", "name": "rsi", "params": {"period": 14}, "tf": "1d"}, "cmp": "<", "rhs": {"kind": "const", "value": 60}}},
            ]},
            {"kind": "cond", "type": "compare", "params": {"lhs": {"kind": "indicator", "name": "rsi", "params": {"period": 14}, "tf": "5m"}, "cmp": ">", "rhs": {"kind": "const", "value": 45}}},
            {"kind": "cond", "type": "compare", "params": {"lhs": {"kind": "price"}, "cmp": ">", "rhs": {"kind": "indicator", "name": "vwap", "tf": "1m"}}},
            {"kind": "cond", "type": "vix", "params": {"cmp": "<", "value": 45}},
            {"kind": "cond", "type": "gap", "params": {"cmp": ">", "value": -3}},
            {"kind": "cond", "type": "price_move", "params": {"ref": "open", "cmp": "<", "value": 2, "absolute": True}},
        ]},
        "max_entries_per_day": 4, "max_open_positions": 2,
    },
    "exits": {"profit_target": {"enabled": True, "value": 40}, "stop_loss": {"enabled": True, "value": 150}, "time_exit": {"enabled": True, "time": "15:45", "dte_lte": 0}},
}


def cfg(start="2024-02-01", end="2024-04-30"):
    return {"start": start, "end": end, "strategies": [STRATEGY], "execution": {"max_quote_age_seconds": 3600}}


def entry_sig(t):
    return (t["entry_ts"], tuple((l["strike"], l["right"], l["expiration"], l["entry_price"]) for l in t["legs"]), t["units"])


def full_sig(t):
    return entry_sig(t) + (t["exit_ts"], round(t["pnl"], 6), t["exit_reason"])


@pytest.mark.parametrize("cut_day,cut_time", [(date(2024, 2, 14), "11:00"), (date(2024, 3, 11), "10:00"), (date(2024, 4, 3), "14:00")])
def test_no_lookahead(synth_store, cut_day, cut_time):
    h, m = map(int, cut_time.split(":"))
    cut = int(datetime(cut_day.year, cut_day.month, cut_day.day, h, m, tzinfo=cal.ET).timestamp())
    base = run_backtest(cfg(), store=synth_store, record_paths=False)
    pert = run_backtest(cfg(), store=PerturbedStore(synth_store, cut), record_paths=False)
    b_entries = [entry_sig(t) for t in base.trades if t["entry_ts"] <= cut]
    p_entries = [entry_sig(t) for t in pert.trades if t["entry_ts"] <= cut]
    assert len(b_entries) > 5
    assert b_entries == p_entries
    b_closed = [full_sig(t) for t in base.trades if t["exit_ts"] <= cut]
    p_closed = [full_sig(t) for t in pert.trades if t["exit_ts"] <= cut]
    assert b_closed == p_closed
    # sanity: the perturbation does change what happens afterwards
    assert [full_sig(t) for t in base.trades if t["entry_ts"] > cut] != [full_sig(t) for t in pert.trades if t["entry_ts"] > cut]


def test_reproducible(synth_store):
    a = run_backtest(cfg(), store=synth_store)
    b = run_backtest(cfg(), store=MarketStore(synth_store.root))
    assert a.fingerprint() == b.fingerprint()
    assert a.meta["config_hash"] == b.meta["config_hash"]
    assert a.meta["dataset_versions"] == b.meta["dataset_versions"]
    assert a.meta["synthetic"] is True


def test_entry_times_survive_dst_and_holidays(synth_store):
    s = dict(STRATEGY, entry={"conditions": {"kind": "group", "children": [{"kind": "cond", "type": "entry_times", "params": {"times": "10:00"}}]}})
    res = run_backtest({"start": "2024-03-04", "end": "2024-04-05", "strategies": [s]}, store=synth_store, record_paths=False)
    assert res.trades
    assert {t["entry_time"][11:16] for t in res.trades} == {"10:00"}
    days = {t["entry_date"] for t in res.trades}
    assert "2024-03-29" not in days  # Good Friday
    assert "2024-03-11" in days or "2024-03-12" in days  # trading continues across the DST change
    offsets = {t["entry_time"][-6:] for t in res.trades}
    assert offsets == {"-05:00", "-04:00"}


def test_accounting_identity_on_synthetic(synth_store):
    res = run_backtest(cfg(), store=synth_store, record_paths=False)
    assert res.daily["equity"][-1] == pytest.approx(100_000 + sum(t["pnl"] for t in res.trades), abs=1e-6)
    assert res.meta["data_quality"]["missing_sessions"] >= 0
