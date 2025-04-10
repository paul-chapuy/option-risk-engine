from typing import Final, Dict
from enum import Enum

from internal.infra.fred import TreasuryYieldID


class Tenor(Enum):
    ONE_MONTH = 1 / 12
    THREE_MONTH = 3 / 12
    SIX_MONTH = 6 / 12
    ONE_YEAR = 1
    TWO_YEAR = 2
    THREE_YEAR = 3
    FIVE_YEAR = 5
    SEVEN_YEAR = 7
    TEN_YEAR = 10
    TWENTY_YEAR = 20
    THIRTY_YEAR = 30


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
