from math import exp, log
from datetime import date
from typing import List

from internal.domain.assets.option import OptionChains, ExcerciceStyle, OptionType
from internal.domain.market.yield_curve import YieldCurve, YieldPoint, YieldCurveType
from internal.modeling.option_pricer import OptionPricer
from internal.modeling.yield_curve import YieldCurveModel


def compute_eu_dividend_yield(
    call_price: float, put_price: float, spot: float, strike: float, T: float, r: float
) -> float:
    """Dividend yield from put-call parity (European options)."""
    DF = exp(-r * T)
    F = (call_price - put_price) / DF + strike
    return r - (1 / T) * log(F / spot)


def compute_am_dividend_yield(
    call_price: float, put_price: float, spot: float, strike: float, T: float, r: float
) -> float:
    q = compute_eu_dividend_yield(call_price, put_price, spot, strike, T, r)

    am_pricer = OptionPricer.make(ExcerciceStyle.American, spot, strike, r, T)
    eu_pricer = OptionPricer.make(ExcerciceStyle.European, spot, strike, r, T)

    while True:
        guess_vol = eu_pricer.implied_volatility(OptionType.Call, call_price, q)

        iv_call = am_pricer.implied_volatility(
            OptionType.Call, call_price, q, guess_vol
        )
        iv_put = am_pricer.implied_volatility(OptionType.Put, put_price, q, guess_vol)

        eu_call_price = eu_pricer.price(OptionType.Call, iv_call, q)
        eu_put_price = eu_pricer.price(OptionType.Put, iv_put, q)

        q_new = compute_eu_dividend_yield(
            eu_call_price, eu_put_price, spot, strike, T, r
        )

        if abs(q_new - q) < 1e-9:
            return q_new

        q = q_new


def build_dividend_curve(
    option_chains: OptionChains,
    interpolated_spot_curve: YieldCurveModel,
    as_of: date,
    min_year_fraction: float,
) -> YieldCurve:

    spot = option_chains.underlying.last_close_price

    points: List[YieldPoint] = []
    for chain in option_chains.chains:
        expiry = chain.expiry
        T = (expiry - as_of).days / 252.0
        if T < min_year_fraction:
            continue

        atm_pair = chain.atm_pair()
        if not atm_pair:
            # TODO: add logs here
            continue

        strike, call, put = atm_pair
        call_price, put_price = call.last_price, put.last_price

        if min(abs(call_price), abs(put_price)) < 1e-3:
            # TODO: add logs here
            continue

        r = interpolated_spot_curve.value(T)

        if option_chains.exercice_style == ExcerciceStyle.American:
            q = compute_am_dividend_yield(call_price, put_price, spot, strike, T, r)
        else:
            q = compute_eu_dividend_yield(call_price, put_price, spot, strike, T, r)

        points.append(YieldPoint(T, q))

    return YieldCurve(points, YieldCurveType.Dividend, as_of)
