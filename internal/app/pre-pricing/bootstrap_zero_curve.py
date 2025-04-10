import os
from typing import Optional

from internal.infra.fred import FredClient, TreasuryYieldID
from internal.domain.market.tenors import TREASURY_YIELD_ID_TO_TENOR
from internal.domain.market.yield_term_structure import YieldTermStructure, YieldPoint
from internal.modeling.yield_term_structure import NelsonSiegel

import matplotlib.pyplot as plt
import numpy as np


def get_fred_ytm_term_structure(api_key: Optional[str] = None) -> YieldTermStructure:
    api_key = api_key or os.environ["FRED_API_KEY"]
    fred_client = FredClient.from_api_key(api_key)
    yield_points = []
    for yield_id, year_to_maturity in TREASURY_YIELD_ID_TO_TENOR.items():
        rep = fred_client.get_latest_yield(yield_id)
        if rep.status != 200:
            raise RuntimeError(f"FRED API error for {yield_id}: {rep}")
        yield_points.append(
            YieldPoint(year_to_maturity.value, float(rep.data.get("value")) / 100)
        )
    return YieldTermStructure(yield_points).fit(model=NelsonSiegel)


def main():
    fred_ytm_term_structure = get_fred_ytm_term_structure()

    synthetic_tenors = np.arange((3 / 12), 30, 0.5)
    synthetic_yields = [fred_ytm_term_structure.model(0, t) for t in synthetic_tenors]

    plt.figure(figsize=(8, 4.5))
    plt.scatter(
        fred_ytm_term_structure.tenors(),
        fred_ytm_term_structure.yields(),
        label="empirical",
    )
    plt.plot(synthetic_tenors, synthetic_yields, label="NS fit")
    plt.xlabel("Tenor")
    plt.ylabel("Yield")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
