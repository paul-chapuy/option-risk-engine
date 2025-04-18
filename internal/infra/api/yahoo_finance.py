from typing import Optional
import yfinance as yf


class YahooClient:

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._yf_ticker = yf.Ticker(self.ticker)

    def get_last_price(self) -> float:
        return self._yf_ticker.info["regularMarketPrice"]

    def get_option_chain(self, expiry: str) -> dict:
        option_chain = self._yf_ticker.option_chain(expiry)
        return {"calls": option_chain.calls, "puts": option_chain.puts}

    def get_all_option_chains(self) -> dict:
        expiries = self._yf_ticker.options
        chains = {}
        for expiry in expiries:
            try:
                chain = self._yf_ticker.option_chain(expiry)
                chains[expiry] = {"calls": chain.calls, "puts": chain.puts}
            except Exception as e:
                print(f"Failed to fetch options for {expiry}: {e}")
        return chains
