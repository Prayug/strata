import math

import pytest

from strata.engine.execution import fill_price, liquidity_cap, option_commission
from strata.engine.margin import MLeg, max_profit_loss, requirement
from strata.engine.models import Execution

C, P = True, False


def L(is_call, k, q, mid=0.0):
    return MLeg(is_call, k, q, 100.0, mid)


# ------------------------------------------------------------------ margin
def test_put_credit_spread_requirement_is_width():
    legs = [L(P, 95, -1, 1.0), L(P, 90, +1, 0.5)]
    assert requirement(legs, 0, 100) == pytest.approx(500)
    mp, ml = max_profit_loss(legs, 0, entry_cash=50)
    assert mp == pytest.approx(50) and ml == pytest.approx(450)


def test_iron_condor_holds_one_side():
    legs = [L(P, 95, -1), L(P, 90, +1), L(C, 105, -1), L(C, 110, +1)]
    assert requirement(legs, 0, 100) == pytest.approx(500)
    legs2 = [L(P, 95, -1), L(P, 85, +1), L(C, 105, -1), L(C, 110, +1)]  # unequal wings -> wider side
    assert requirement(legs2, 0, 100) == pytest.approx(1000)


def test_debit_structures_need_no_extra_margin():
    assert requirement([L(C, 100, +1)], 0, 100) == 0
    assert requirement([L(C, 100, +1), L(C, 105, -1)], 0, 100) == 0  # bull call spread
    assert requirement([L(C, 95, +1), L(C, 100, -2), L(C, 105, +1)], 0, 100) == 0  # long butterfly


def test_naked_regt_formulas():
    # put: max(20% * 100 - 5 OTM, 10% * 95) + 1.00 premium = 16 per share
    assert requirement([L(P, 95, -1, 1.0)], 0, 100) == pytest.approx(1600)
    # call: max(20 - 5, 10) + 0.5 = 15.5
    assert requirement([L(C, 105, -1, 0.5)], 0, 100) == pytest.approx(1550)
    # deep OTM put floors at 10% of strike + premium
    assert requirement([L(P, 70, -1, 0.05)], 0, 100) == pytest.approx((7.0 + 0.05) * 100)
    # strangle: greater side + premium of the other side = max(1600+50, 1550+100)
    assert requirement([L(P, 95, -1, 1.0), L(C, 105, -1, 0.5)], 0, 100) == pytest.approx(1650)


def test_cash_model():
    assert requirement([L(P, 95, -1, 1.0)], 0, 100, "cash") == pytest.approx(9500)  # cash secured
    assert math.isinf(requirement([L(C, 105, -1, 0.5)], 0, 100, "cash"))


def test_covered_call_calendar_diagonal():
    assert requirement([L(C, 105, -1, 1.0)], 100, 100) == 0  # covered call
    assert requirement([L(C, 100, -1), L(C, 100, +1)], 0, 100) == 0  # calendar (long leg later)
    assert requirement([L(C, 100, -1), L(C, 105, +1)], 0, 100) == pytest.approx(500)  # diagonal, long higher strike


def test_ratio_spread_uncovered_portion_is_naked():
    # 1x2 put ratio: long 100P, short 2x 95P -> one uncovered 95P margined naked, rest defined
    r = requirement([L(P, 100, +1, 3.0), L(P, 95, -2, 1.0)], 0, 100)
    assert r == pytest.approx(1600)


# ------------------------------------------------------------------ execution
def test_fill_models():
    ex = Execution()
    assert fill_price(1.00, 1.10, +1, ex) == pytest.approx(1.10)
    assert fill_price(1.00, 1.10, -1, ex) == pytest.approx(1.00)
    ex = Execution(fill_model="midpoint")
    assert fill_price(1.00, 1.10, +1, ex) == pytest.approx(1.05)
    ex = Execution(fill_model="custom", custom_fraction=0.5, slippage_per_contract=0.01)
    assert fill_price(1.00, 1.20, +1, ex) == pytest.approx(1.10 + 0.05 + 0.01)
    assert fill_price(1.00, 1.20, -1, ex) == pytest.approx(1.10 - 0.05 - 0.01)
    assert fill_price(0.00, 0.05, -1, Execution(slippage_per_contract=0.02)) == 0.0  # never negative


def test_commissions_and_liquidity():
    ex = Execution(commission_per_contract=0.65, fee_per_contract=0.05)
    assert option_commission(-3, ex) == pytest.approx(2.10)
    ex = Execution(participation_rate=0.5, max_volume_pct=10)
    assert liquidity_cap(40, 1000, ex) == 20
    assert liquidity_cap(400, 100, ex) == 10
    assert math.isinf(liquidity_cap(-1, -1, Execution()))
