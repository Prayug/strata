"""V6: what a budget buys, for the sessions one strategy needs.

Pure arithmetic over the published coverage and prices recorded in `market.OFFERS` (read from the providers' pages on
`market.CHECKED`). Nothing here contacts a provider, and none of it says a provider works.
"""
from __future__ import annotations

from datetime import date

import pytest

from strata.data.providers import market as M

TODAY = date(2026, 11, 25)
DAYS = (["2008-10-07", "2008-10-10"] + ["2010-05-07", "2010-05-21"] + ["2011-08-05", "2011-08-09", "2011-08-11"] + ["2016-06-27"] + ["2018-02-06", "2018-02-09"]
        + ["2020-03-06", "2020-03-10", "2020-03-13", "2020-03-23"] + ["2022-05-06", "2022-06-17"] + ["2025-04-04", "2025-04-07"] + ["2026-11-23"])


@pytest.fixture(autouse=True)
def _own_data_directory(tmp_path):
    """The table says what was MEASURED on this installation when an access report exists. These tests must not read the
    developer's own report: each runs against an empty data directory."""
    from strata.config import get_settings, override_settings
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path / "data")
    yield
    override_settings(data_dir=old)


def need(**kw):
    return {"symbol": "SPY", "days": DAYS, "no_contract_days": ["2020-03-10"], "contracts_per_signal": 13, "sessions_per_signal": 1, "sessions_with_quotes": 6, "sessions_with_trades": 1, "sessions_with_intraday_quotes": 1,
            "contract_existence": {"known_sessions": 8, "unknown_sessions": 11, "no_contract": 1, "note": ""}, **kw}


def offer(bt, oid):
    return next(o for o in bt["offers"] if o["id"] == oid)


def test_each_offer_counts_only_the_sessions_inside_its_history():
    bt = M.budget_table(need(), TODAY)
    assert bt["signal_sessions"] == 19 and bt["testable_sessions"] == 18 and bt["no_contract_sessions"] == 1, "a session with no listed contract is untestable whatever is bought"
    assert offer(bt, "alpaca_free")["sessions"] == 3 and offer(bt, "alpaca_free")["window"][0] == "2024-01-18"  # 2025-04-04, 2025-04-07, 2026-11-23; the start is the first session that returned a bar
    assert offer(bt, "massive_basic")["sessions"] == 3 and offer(bt, "massive_basic")["window"] == ["2024-11-25", "2026-11-24"]
    db_ = offer(bt, "databento_credit")
    assert db_["sessions"] == 11 and db_["window"][0] == "2013-04-01"  # everything from 2016 on except the session with no contract
    # 11 sessions x 13 contracts x 390 one-minute records x 80 bytes at $2 per GB
    assert db_["contract_sessions"] == 143 and db_["estimated_gb"] == pytest.approx(143 * 390 * 80 / 1e9, abs=5e-4) and db_["estimated_cost"] == pytest.approx(0.01, abs=0.005) and db_["credit"] == 125.0
    assert offer(bt, "optionsdx_30m")["by_year"] == {2010: 2, 2011: 3, 2016: 1, 2018: 2, 2020: 3, 2022: 2}, "nothing before 2010 or after 2023"
    assert offer(bt, "optionsdx_15m")["by_year"] == {2011: 3, 2016: 1, 2018: 2, 2020: 3, 2022: 2}
    assert offer(bt, "optionsdx_5m")["by_year"] == {2016: 1, 2018: 2, 2020: 3, 2022: 2} and offer(bt, "optionsdx_1m")["by_year"] == {2020: 3, 2022: 2}
    assert offer(bt, "thetadata_value")["sessions"] == 8 and offer(bt, "thetadata_standard")["sessions"] == 11 and offer(bt, "thetadata_pro")["sessions"] == 11
    assert offer(bt, "massive_advanced")["sessions"] == 5  # quotes start 2022-03-07
    assert all(o["sessions"] <= 18 and o["of"] == 18 for o in bt["offers"])


def test_a_budget_buys_the_years_with_the_most_signals_first_and_never_more_than_it_allows():
    bt = M.budget_table(need(), TODAY)
    rows = {r["budget"]: r for r in bt["budgets"]}
    assert list(rows) == [0, 25, 50, 100, 200]
    zero = rows[0]["best"]
    assert zero["cost"] == 0 and zero["sessions_with_bid_ask"] == 11 and zero["sessions_trade_prints_only"] == 0 and zero["resolution_minutes"] == 1
    assert {i["offer"] for i in zero["items"]} == {"alpaca_free", "massive_basic", "databento_credit"} and len(zero["needs"]) == 3
    for b, r in rows.items():
        for plan in [r["best"]] + r["alternatives"]:
            assert plan["cost"] <= b and plan["sessions"] <= 18 and plan["sessions_with_bid_ask"] + plan["sessions_trade_prints_only"] == plan["sessions"]
    b25 = rows[25]["best"]
    assert b25["label"].startswith("OptionsDX 30-minute chains: 2010-2011, 2016, 2018, 2020, 2022") and b25["cost"] == 18.0, "every year with a signal fits: six years at $3"
    assert b25["sessions_with_bid_ask"] == 16 and b25["first"] == "2010-05-07", "the 5 sessions of 2010-2011 are added to the free 11"
    one_min = {g["granularity"]: g for g in rows[200]["by_granularity"]}
    assert one_min["1-minute"]["sessions_with_bid_ask"] == 11 and one_min["30-minute or finer"]["sessions_with_bid_ask"] == 16
    # a 5-minute year costs $20: one year fits in $25, two in $50
    five = next(a for a in [rows[25]["best"]] + rows[25]["alternatives"] if "OptionsDX 5-minute" in a["label"])
    assert five["cost"] == 20.0 and five["items"][0]["years"] == [2020]
    five50 = [a for a in [rows[50]["best"]] + rows[50]["alternatives"] if "OptionsDX 5-minute" in a["label"]]
    assert not five50 or five50[0]["items"][0]["years"] == [2020, 2022]


def test_without_the_free_accounts_nothing_is_free_and_the_table_says_what_each_needs():
    n = need(days=["2008-10-07", "2010-05-07"], no_contract_days=[])
    bt = M.budget_table(n, TODAY)
    rows = {r["budget"]: r for r in bt["budgets"]}
    assert rows[0]["nothing"] is True and rows[0]["best"] is None, "no free source reaches 2008 or 2010"
    assert rows[25]["best"]["label"].startswith("OptionsDX 30-minute chains: 2010") and rows[25]["best"]["sessions"] == 1 and rows[25]["best"]["of"] == 2
    assert any("Algo Trader Plus" in x for x in bt["not_offers"]) and any("End-of-day" in x for x in bt["not_offers"])
    assert "never buys anything" in " ".join(bt["notes"]) and bt["checked"] == M.CHECKED


def test_a_symbol_the_vendor_does_not_sell_is_not_counted():
    bt = M.budget_table(need(symbol="IWM"), TODAY)
    assert offer(bt, "optionsdx_30m")["sessions"] == 0 and offer(bt, "optionsdx_30m")["window"] is None and "No IWM product" in offer(bt, "optionsdx_30m")["caveat"]
    assert offer(bt, "databento_credit")["sessions"] == 11
    q = M.budget_table(need(symbol="QQQ"), TODAY)
    assert offer(q, "optionsdx_30m")["by_year"] == {2016: 1, 2018: 2, 2020: 3, 2022: 2} and offer(q, "optionsdx_1m")["sessions"] == 0, "QQQ starts in 2012 and has no 1-minute files"


def test_the_catalogue_states_what_was_verified_and_what_was_not():
    by = {e["id"]: e for e in M.ENTRIES}
    assert by["optionsdx"]["tested"].startswith("verified on real data") and "A purchased full-session file was not tested" in by["optionsdx"]["tested"]
    assert "401 without a key" in by["alpaca"]["tested"]
    for k in ("databento", "thetadata", "massive"):
        assert "never run against the live service" in by[k]["tested"]
    assert "NOT in the list" in " ".join(by["wrds_optionmetrics"]["limits"]) and "End of day only" in " ".join(by["wrds_optionmetrics"]["limits"])
    assert by["uw_wrds"]["integration"] == "import" and "not allow systematic downloading" in " ".join(by["uw_wrds"]["limits"]) and by["uw_bloomberg"]["integration"] == "none"
    assert all(o["entry"] in by for o in M.OFFERS) and {o["observation"] for o in M.OFFERS} == {"trade prints", "bid/ask quotes"}


def test_sessions_before_every_source_are_named_not_buried_in_the_denominator():
    """Two of these sessions are in 2008: no source compared here reaches them, at any price. Saying "N of 18" alone would
    make every source look worse than it is and hide that the early years cannot be bought."""
    bt = M.budget_table(need(), TODAY)
    assert bt["earliest_source_date"] == "2010-01-01" and bt["before_any_source"] == 2 and bt["reachable_sessions"] == 16
    assert "2 of the 18 sessions fall before 2010-01-01" in bt["denominator_note"] and "no budget buys them" in bt["denominator_note"]
    assert "established for 8 of the 19 sessions only" in bt["denominator_note"] and "upper bound" in bt["denominator_note"]
    best = max((r["best"] for r in bt["budgets"] if r["best"]), key=lambda b: b["sessions"])
    assert best["sessions"] <= bt["reachable_sessions"], "no budget reaches a session before the earliest source"
    recent = M.budget_table(need(days=["2025-04-04", "2026-11-23"], no_contract_days=[]), TODAY)
    assert recent["before_any_source"] == 0 and recent["reachable_sessions"] == 2 and "fall before" not in recent["denominator_note"]


def test_a_source_is_described_by_what_was_measured_once_its_key_has_been_used(tmp_path, monkeypatch):
    """Until the access test has returned an observation with the stored key, Alpaca's row says it was never run. Afterwards
    it states the measurement and its date, and the other key-gated routes are still described as unrun."""
    import json

    from strata.config import get_settings, override_settings
    old = get_settings().data_dir
    override_settings(data_dir=tmp_path)
    try:
        bt = M.budget_table(need(), TODAY)
        row = offer(bt, "alpaca_free")
        assert "verified" not in row and "Never run with a real key" in row["caveat"]
        assert any("None of the paid or key-gated routes has been run" in n for n in bt["notes"])
        # a report without a single observation changes nothing: authentication alone establishes no coverage
        (tmp_path / "access").mkdir()
        (tmp_path / "access" / "alpaca.json").write_text(json.dumps({"provider": "alpaca", "status": "ran", "verified": False, "run_at": "2026-11-26T05:17:28+00:00", "matrix": [{"date": "2024-02-14", "contracts": [{"has_observations": False}]}]}))
        assert "verified" not in offer(M.budget_table(need(), TODAY), "alpaca_free")
        (tmp_path / "access" / "alpaca.json").write_text(json.dumps({"provider": "alpaca", "status": "ran", "verified": True, "run_at": "2026-11-26T05:17:28+00:00", "findings": {"earliest_option_bars": "2024-01-18"},
                                                                      "matrix": [{"date": "2024-02-14", "contracts": [{"has_observations": True}] * 9 + [{"has_observations": False}]}, {"date": "2026-11-18", "contracts": [{"has_observations": True}] * 10}]}))
        bt = M.budget_table(need(), TODAY)
        row = offer(bt, "alpaca_free")
        assert row["verified"] is True and "Measured with the key stored here on 2026-11-26: 19 of 20 expired contract-sessions" in row["caveat"] and "2024-02-14 to 2026-11-18" in row["caveat"]
        assert "Never run" not in row["caveat"] and "no bid/ask" in row["caveat"], "what the data is does not change: trade prints"
        assert "Never run with a real key" in offer(bt, "massive_basic")["caveat"] and "Never run with a real key" in offer(bt, "databento_credit")["caveat"]
        assert any("Alpaca's coverage was measured with the key stored here (2026-11-26)" in n and "other paid or key-gated routes have not been run" in n for n in bt["notes"])
        assert row["sessions"] == 3, "the count of this strategy's sessions inside the history is the same: a measurement of access is not a promise about every contract"
    finally:
        override_settings(data_dir=old)
