from enum import Enum
from typing import Optional
from datetime import date

from internal.infra.api import Response, Context, API, QueryParamAuth


class TreasuryYield(Enum):
    """Daily Treasury Constant Maturity Yields from FRED"""

    ONE_MONTH = "DGS1MO"
    THREE_MONTH = "DGS3M"
    SIX_MONTH = "DGS6MO"
    ONE_YEAR = "DGS1"
    TWO_YEAR = "DGS2"
    THREE_YEAR = "DGS3"
    FIVE_YEAR = "DGS5"
    SEVEN_YEAR = "DGS7"
    TEN_YEAR = "DGS10"
    TWENTY_YEAR = "DGS20"
    THIRTY_YEAR = "DGS30"

    @classmethod
    def values(cls):
        return [member.value for member in cls]


class FredClient:

    _url = "https://api.stlouisfed.org/fred/"

    def __init__(self, ctx: Context) -> None:
        self._api = API(ctx)

    @staticmethod
    def from_api_key(api_key: str) -> "FredClient":
        auth = QueryParamAuth(api_key)
        ctx = Context.make(auth)
        return FredClient(ctx)

    def get_treasury_yield_series(
        self,
        treasury_yield: TreasuryYield,
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
