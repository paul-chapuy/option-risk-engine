from math import exp, log
from datetime import date, timedelta
from typing import List

from internal.domain.assets.option import OptionChains, ExcerciceStyle, OptionType
from internal.domain.market.volatility_surface import IVSurface, IVSlice, IVPoint
from internal.modeling.option_pricer import OptionPricer
from internal.modeling.yield_curve import YieldCurveModel


def compute_moneyness(S: float, K: float) -> float:
    return log(S / K)


def compute_forward_moneyness(
    S: float, K: float, r: float, q: float, T: float
) -> float:
    F = S * exp((r - q) * T)
    return log(F / K)


def compute_volatility_surface(
    option_chains: OptionChains,
    interp_spot_curve: YieldCurveModel,
    interp_divivend_curve: YieldCurveModel,
    as_of: date,
) -> IVSurface:
    ex_style = option_chains.exercice_style
    S = option_chains.underlying.last_close_price

    slices: List[IVSlice] = []
    for chain in option_chains.chains:
        T = (chain.expiry - as_of).days / 252.0
        if T <= 0.0:
            continue

        r = interp_spot_curve.value(T)
        q = interp_divivend_curve.value(T)

        slice: List[IVPoint] = []
        for opt in chain.options:

            if (
                opt.volume < 10
                or opt.last_price < 0.1
                or opt.last_trade_date.date() < (as_of - timedelta(days=1))
            ):
                continue

            K = opt.strike
            moneyness = compute_moneyness(S, K)
            fwd_moneyness = compute_forward_moneyness(S, K, r, q, T)

            # don't calibrate on ITM options
            if (opt.option_type == OptionType.Call) and (fwd_moneyness > 0.0):
                continue

            if (opt.option_type == OptionType.Put) and (fwd_moneyness < 0.0):
                continue

            pricer: OptionPricer = OptionPricer.make(ex_style, S, K, r, T)
            iv = pricer.implied_volatility(opt.option_type, opt.last_price, q)

            slice.append(
                IVPoint(
                    year_to_maturity=T,
                    strike=K,
                    moneyness=moneyness,
                    forward_moneyness=fwd_moneyness,
                    value=iv,
                )
            )

        slices.append(IVSlice(T, slice))

    return IVSurface(slices)
