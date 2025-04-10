from enum import Enum
from typing import Optional
from datetime import date, timedelta

from internal.infra.api import Response, Context, API, QueryParamAuth


class TreasuryYieldID(Enum):
    """Daily Treasury Constant Maturity Yields from FRED"""

    ONE_MONTH = "DGS1MO"
    SIX_MONTH = "DGS6MO"
    ONE_YEAR = "DGS1"
    TWO_YEAR = "DGS2"
    THREE_YEAR = "DGS3"
    FIVE_YEAR = "DGS5"
    SEVEN_YEAR = "DGS7"
    TEN_YEAR = "DGS10"
    TWENTY_YEAR = "DGS20"
    THIRTY_YEAR = "DGS30"


class FredClient:

    _url = "https://api.stlouisfed.org/fred/"

    def __init__(self, ctx: Context) -> None:
        self._api = API(ctx)

    @staticmethod
    def from_api_key(api_key: str) -> "FredClient":
        auth = QueryParamAuth(api_key)
        ctx = Context.make(auth)
        return FredClient(ctx)

    def get_yield_series(
        self,
        treasury_yield: TreasuryYieldID,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> Response:
        params = {
            "series_id": treasury_yield.value,
        }

        if start is not None:
            params["observation_start"] = start.isoformat()
        if end is not None:
            params["observation_end"] = end.isoformat()

        return self._api.request(
            method="GET",
            url=(self._url, "series/observations"),
            params=params,
        )

    def get_latest_yield(
        self, treasury_yield: TreasuryYieldID, max_days_back: int = 7
    ) -> Response:
        today = date.today()

        for delta in range(max_days_back):
            target_date = today - timedelta(days=delta)
            response = self.get_yield_series(
                treasury_yield,
                start=target_date,
                end=target_date,
            )

            if response.status == 200 and response.data.get("observations"):
                obs = response.data["observations"][0]

                if obs.get("value") not in ("", "."):
                    return Response(data=obs, status=200)

        return Response(data=None, status=404)
