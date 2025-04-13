import os
from typing import Optional, Final, Dict, List
from math import log, exp
from internal.infra.fred import FredClient, TreasuryYieldID
from internal.domain.market.yield_term_structure import (
    Tenor,
    YieldTermStructure,
    YieldPoint,
    TermStructureType,
)
from internal.modeling.yield_term_structure import NelsonSiegel

import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

TREASURY_YIELD_ID_TO_TENOR: Final[Dict[TreasuryYieldID, Tenor]] = {
    TreasuryYieldID.ONE_MONTH: Tenor.ONE_MONTH,
    TreasuryYieldID.SIX_MONTH: Tenor.SIX_MONTH,
    TreasuryYieldID.ONE_YEAR: Tenor.ONE_YEAR,
    TreasuryYieldID.TWO_YEAR: Tenor.TWO_YEAR,
    TreasuryYieldID.THREE_YEAR: Tenor.THREE_YEAR,
    TreasuryYieldID.FIVE_YEAR: Tenor.FIVE_YEAR,
    TreasuryYieldID.SEVEN_YEAR: Tenor.SEVEN_YEAR,
    TreasuryYieldID.TEN_YEAR: Tenor.TEN_YEAR,
    TreasuryYieldID.TWENTY_YEAR: Tenor.TWENTY_YEAR,
    TreasuryYieldID.THIRTY_YEAR: Tenor.THIRTY_YEAR,
}


def get_ytm_term_structure(api_key: Optional[str] = None) -> YieldTermStructure:
    api_key = api_key or os.environ["FRED_API_KEY"]
    fred_client = FredClient.from_api_key(api_key)

    def fetch_yield_point(yield_id: TreasuryYieldID, tenor: Tenor) -> YieldPoint:
        response = fred_client.get_latest_yield(yield_id)
        if response.status != 200:
            raise RuntimeError(f"FRED API error for {yield_id}: {response}")
        value = float(response.data["value"]) / 100
        return YieldPoint(tenor.value, value)

    yield_points = [
        fetch_yield_point(yield_id, tenor)
        for yield_id, tenor in TREASURY_YIELD_ID_TO_TENOR.items()
    ]

    return YieldTermStructure.make(yield_points, TermStructureType.YTM, NelsonSiegel)


def bootstrap_spot_term_structure(
    ytm_term_scructure: YieldTermStructure,
) -> YieldTermStructure:
    print(ytm_term_scructure.points[0].yield_rate)

    spot_points = []
    dt = 0.5
    while dt <= 30.0:
        semi_annual_rate = ytm_term_scructure.model(0, dt)
        coupon = semi_annual_rate / 2
        previous_coupon_value = 0

        for p in spot_points:
            if p.year_to_maturity >= dt:
                break
            _dt, _r = p.year_to_maturity, p.yield_rate
            previous_coupon_value += coupon * exp(-_r * _dt)

        spot_rate = -log((1 - previous_coupon_value) / (1 + coupon)) / dt
        spot_points.append(YieldPoint(dt, spot_rate))
        dt += 0.5

    dt_initial = 1 / 12
    spot_rate_initial = 12 * log(1 + ytm_term_scructure.points[0].yield_rate / 12)
    spot_points.append(YieldPoint(dt_initial, spot_rate_initial))

    return YieldTermStructure.make(spot_points, TermStructureType.Spot, NelsonSiegel)


def price_bond_test(
    maturity: float, spot_ts: YieldTermStructure, ytm_ts: YieldTermStructure
) -> float:
    coupon = float(ytm_ts.model(0, maturity) / 2)
    cashflows = [coupon] * int(maturity / 0.5)
    cashflows[-1] += 1
    times = [0.5 * (i + 1) for i in range(len(cashflows))]
    npv = 0
    for i, (t, cf) in enumerate(zip(times, cashflows)):
        r = spot_ts.model(0, t)
        npv += cf * exp(-r * t)
    return npv


def main():
    ytm_term_structure = get_ytm_term_structure()
    spot_term_stucture = bootstrap_spot_term_structure(ytm_term_structure)

    for y in range(1, 31, 1):
        npv = price_bond_test(y, spot_term_stucture, ytm_term_structure)
        print(y, npv)
    # a = generate_coupon_schedule(
    #     date.today() + relativedelta(years=30),
    #     date.today(),
    # )
    # print(a)
    # print("YTM")
    # for t in spot_term_stucture.tenors():
    #     print(t, ytm_term_structure.model(0, t))

    # print("SPOT")
    # for t in spot_term_stucture.tenors():
    #     print(t, spot_term_stucture.model(0, t))

    synthetic_tenors = np.arange((1 / 12), 30, 0.1)
    synthetic_yields = [ytm_term_structure.model(0, t) for t in synthetic_tenors]
    # print(ytm_term_structure.model(0, 1 / 365))
    # print(ytm_term_structure.model(0, 1 / 12))
    # print(ytm_term_structure.model(0, 6 / 12))
    # print(ytm_term_structure.model(0, 30))

    plt.figure(figsize=(8, 4.5))
    plt.scatter(
        ytm_term_structure.tenors(),
        ytm_term_structure.yields(),
        label="empirical ytm",
    )
    plt.scatter(
        spot_term_stucture.tenors(),
        spot_term_stucture.yields(),
        label="empirical spot",
    )
    plt.plot(synthetic_tenors, synthetic_yields, label="NS fit")
    plt.xlabel("Tenor")
    plt.ylabel("Yield")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
