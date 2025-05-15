from dataclasses import dataclass
from enum import Enum


class Ticker(Enum):
    SPY = "SPY"
    AAPL = "AAPL"
    AMZN = "AMZN"
    TSLA = "TSLA"
    MSFT = "MSFT"
    SPX = "^SPX"


@dataclass
class Equity:
    ticker: Ticker
    last_price: float
    last_close_price: float
