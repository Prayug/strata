"""Fill-price, commission and liquidity models.

conservative (default): buy at the ask, sell at the bid
midpoint:               fill at (bid + ask) / 2
custom:                 mid +/- fraction * half-spread against you (fraction 1 == conservative)
Every model then adds `slippage_per_contract` against you. Prices are never better than
the quoted midpoint for the side being traded and never negative.
"""
from __future__ import annotations

import math

from strata.engine.models import Execution


def fill_price(bid: float, ask: float, side: int, ex: Execution) -> float:
    """side +1 = buy, -1 = sell. Returns price per share."""
    mid = 0.5 * (bid + ask)
    if ex.fill_model == "conservative":
        p = ask if side > 0 else bid
    elif ex.fill_model == "midpoint":
        p = mid
    else:
        p = mid + side * ex.custom_fraction * 0.5 * (ask - bid)
    p += side * ex.slippage_per_contract
    return max(0.0, p)


def option_commission(contracts: int, ex: Execution) -> float:
    return abs(contracts) * (ex.commission_per_contract + ex.fee_per_contract)


def stock_fill_price(price: float, side: int, ex: Execution) -> float:
    return price * (1 + side * ex.stock_slippage_bps / 10_000.0)


def stock_commission(shares: float, ex: Execution) -> float:
    return abs(shares) * ex.stock_commission_per_share


def liquidity_cap(size: int, volume: int, ex: Execution) -> float:
    """Max contracts fillable for one leg at this quote. Unknown size/volume (-1) = no cap."""
    cap = math.inf
    if size >= 0:
        cap = math.floor(size * ex.participation_rate)
    if ex.max_volume_pct is not None and volume >= 0:
        cap = min(cap, math.floor(volume * ex.max_volume_pct / 100.0))
    return cap
