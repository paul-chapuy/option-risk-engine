from typing import Dict, List, Tuple
import yfinance as yf

from pandas import DataFrame


class YahooClient:

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._yf_ticker = yf.Ticker(self.ticker)

    def get_last_price(self) -> float:
        return self._yf_ticker.info.get("regularMarketPrice")

    def get_option_chain(self, expiry: str) -> Tuple[DataFrame | DataFrame]:
        option_chain = self._yf_ticker.option_chain(expiry)
        return option_chain.calls, option_chain.puts

    def get_options_expirires(self) -> Tuple[str]:
        return self._yf_ticker.options
