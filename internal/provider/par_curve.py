import os
import pickle
from pathlib import Path
from typing import Final, Dict, List, Optional
from datetime import datetime, date

from internal.infra.api.fred import FredClient, TreasuryYieldID
from internal.domain.market.yield_curve import YieldCurve, YieldPoint, YieldCurveType


class ParCurveProvider:

    def __init__(self, snapshot: Optional[str] = None, api_key: Optional[str] = None):
        self.snapshot = snapshot or date.today().strftime("%Y-%m-%d")
        self.cache_path = (
            Path(__file__).resolve().parents[2] / f"cache/par_curve_{self.snapshot}.pkl"
        )
        self.api_key = api_key or os.environ["FRED_API_KEY"]

    def get(self) -> YieldCurve:
        try:
            curve = self.get_from_cache()
            return curve
        except Exception as e:
            curve = self.get_from_fred()
            self.save_to_cache(curve)
            return curve

    def get_from_fred(self) -> YieldCurve:
        return YieldCurve(self._get_yield_points_from_fred(), YieldCurveType.Par)

    def get_from_cache(self) -> YieldCurve:
        if self.cache_path is None or not self.cache_path.exists():
            raise FileNotFoundError(
                "No valid cache path defined or file does not exist."
            )

        try:
            with open(self.cache_path, "rb") as f:
                curve = pickle.load(f)
                if not isinstance(curve, YieldCurve):
                    raise TypeError("Cache does not contain a valid YieldCurve object.")
                return curve

        except Exception as e:
            raise RuntimeError(f"Failed to load yield curve from cache: {e}")

    def _get_yield_points_from_fred(self) -> List[YieldPoint]:
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

        fred_client = FredClient.from_api_key(self.api_key)

        yield_points = []
        for yield_id, tenor_yf in TREASURY_YIELD_ID_TO_TENOR.items():
            response = fred_client.get_latest_yield(yield_id)
            if response.status != 200:
                raise RuntimeError(f"FRED API error for {yield_id}: {response}")

            rate = float(response.data["value"]) / 100
            eval_date = datetime.strptime(response.data["date"], "%Y-%m-%d").date()
            yield_points.append(YieldPoint(tenor_yf, rate, eval_date))

        return yield_points

    def save_to_cache(self, curve: YieldCurve) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(curve, f)
            print(f"Curve cached at: {self.cache_path}")
