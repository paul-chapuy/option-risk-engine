from math import exp, log
from typing import Final, List, Tuple

from internal.domain.market.yield_curve import YieldCurve, YieldCurveType, YieldPoint
from internal.modeling.yield_curve import NelsonSiegel


def compute_spot_rate(dt: float, spot_points: List[YieldPoint], coupon: float) -> float:
    pv_coupons = sum(
        coupon * exp(-p.value * p.year_to_maturity)
        for p in spot_points
        if p.year_to_maturity < dt
    )
    return -log((1 - pv_coupons) / (1 + coupon)) / dt


def compute_one_month_spot_rate(par_curve: YieldCurve) -> Tuple[float, float]:
    one_month_yf: Final[float] = 1 / 12
    r = par_curve.points[0].value
    spot_rate = log(1 + r * one_month_yf) / one_month_yf
    return one_month_yf, spot_rate


def bootstrap_spot_curve(
    par_curve: YieldCurve, max_tenor: float = 30.0, step: float = 0.5
) -> YieldCurve:

    spot_points = []
    evaluation_date = par_curve.evaluation_date

    fitted_par_curve = par_curve.calibrate(NelsonSiegel)
    dt = step
    while dt <= max_tenor:
        semi_annual_rate = fitted_par_curve.value(dt)
        coupon = semi_annual_rate / 2
        spot_rate = compute_spot_rate(dt, spot_points, coupon)

        spot_points.append(YieldPoint(dt, spot_rate, evaluation_date))
        dt += step

    dt_initial, spot_rate_initial = compute_one_month_spot_rate(par_curve)
    spot_points.append(YieldPoint(dt_initial, spot_rate_initial, evaluation_date))

    return YieldCurve(
        spot_points,
        yield_type=YieldCurveType.Spot,
    )
