import os
from typing import Optional, Final, Dict, List
from math import log, exp
from datetime import datetime

from internal.infra.api.fred import FredClient, TreasuryYieldID
from internal.domain.market.yield_term_structure import (
    YieldTermStructure,
    YieldPoint,
    TermStructureType,
)
from internal.modeling.yield_term_structure import NelsonSiegel

import matplotlib.pyplot as plt
import numpy as np


TREASURY_YIELD_ID_TO_TENOR: Final[Dict[TreasuryYieldID, float]] = {
    TreasuryYieldID.ONE_MONTH: 1 / 12,
    TreasuryYieldID.SIX_MONTH: 6 / 12,
    TreasuryYieldID.ONE_YEAR: 1.0,
    TreasuryYieldID.TWO_YEAR: 2.0,
    TreasuryYieldID.THREE_YEAR: 3.0,
    TreasuryYieldID.FIVE_YEAR: 5.0,
    TreasuryYieldID.SEVEN_YEAR: 7.0,
    TreasuryYieldID.TEN_YEAR: 10.0,
    TreasuryYieldID.TWENTY_YEAR: 20.0,
    TreasuryYieldID.THIRTY_YEAR: 30.0,
}


def fetch_latest_ytm_yield_points(fred_client: FredClient) -> List[YieldPoint]:
    yield_points = []

    for yield_id, tenor in TREASURY_YIELD_ID_TO_TENOR.items():
        response = fred_client.get_latest_yield(yield_id)
        if response.status != 200:
            raise RuntimeError(f"FRED API error for {yield_id}: {response}")

        rate = float(response.data["value"]) / 100
        eval_date = datetime.strptime(response.data["date"], "%Y-%m-%d").date()
        yield_points.append(YieldPoint(tenor, rate, eval_date))

    return yield_points


def get_ytm_term_structure(api_key: Optional[str] = None) -> YieldTermStructure:
    api_key = api_key or os.environ["FRED_API_KEY"]
    fred_client = FredClient.from_api_key(api_key)
    yield_points = fetch_latest_ytm_yield_points(fred_client)

    return YieldTermStructure.from_raw_data(
        yield_points,
        term_structure_type=TermStructureType.YTM,
        model=NelsonSiegel,
    )


def _bootstrap_spot_rate(
    dt: float, spot_points: List[YieldPoint], coupon: float
) -> float:
    pv_coupons = sum(
        coupon * exp(-p.yield_rate * p.year_to_maturity)
        for p in spot_points
        if p.year_to_maturity < dt
    )
    return -log((1 - pv_coupons) / (1 + coupon)) / dt


def bootstrap_spot_term_structure(
    ytm_term_structure: YieldTermStructure, max_tenor: float = 30.0, step: float = 0.5
) -> YieldTermStructure:

    evaluation_date = ytm_term_structure.evaluation_date
    spot_points = []

    dt = step
    while dt <= max_tenor:
        semi_annual_rate = ytm_term_structure.model(dt)
        coupon = semi_annual_rate / 2
        spot_rate = _bootstrap_spot_rate(dt, spot_points, coupon)

        spot_points.append(YieldPoint(dt, spot_rate, evaluation_date))
        dt += step

    # 1M tenor handled as ZCB with discrete-to-continuous conversion
    dt_initial = 1 / 12
    r_initial = ytm_term_structure.points[0].yield_rate
    spot_rate_initial = 12 * log(1 + r_initial / 12)
    spot_points.append(YieldPoint(dt_initial, spot_rate_initial, evaluation_date))

    return YieldTermStructure.from_raw_data(
        spot_points,
        term_structure_type=TermStructureType.Spot,
        model=NelsonSiegel,
    )


def main():
    ytm_term_structure = get_ytm_term_structure()
    spot_term_structure = bootstrap_spot_term_structure(ytm_term_structure)
    # we now have the spot curve that we can store somewhere and re-use for calibrate vol surface & pricing

    synthetic_tenors = np.arange((1 / 12), 30, 0.1)
    synthetic_fred_ytm = [ytm_term_structure.model(dt) for dt in synthetic_tenors]
    synthetic_spot_curve = [spot_term_structure.model(dt) for dt in synthetic_tenors]

    plt.figure(figsize=(8, 4.5))
    plt.scatter(
        ytm_term_structure.tenors(),
        ytm_term_structure.yields(),
        label="empirical ytm",
    )
    plt.plot(synthetic_tenors, synthetic_fred_ytm, label="NS fit fred ytm")

    plt.scatter(
        spot_term_structure.tenors(),
        spot_term_structure.yields(),
        label="empirical spot",
    )
    plt.plot(synthetic_tenors, synthetic_spot_curve, label="NS fit spot curve")

    plt.legend()
    plt.xlabel("Tenor")
    plt.ylabel("Yield")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
