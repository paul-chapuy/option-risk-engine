import os
import pickle
from pathlib import Path
from typing import Optional
from datetime import date

from internal.domain.market.option_chains import OptionChains
from internal.infra.api.yahoo_finance import YahooClient

import pandas as pd


class OptionChainsProvider:

    def __init__(self, ticker: str, snapshot: Optional[str] = None):
        self.ticker = ticker
        self.snapshot = snapshot or date.today().strftime("%Y-%m-%d")
        self.cache_path = (
            Path(__file__).resolve().parents[2]
            / f"cache/option_chains_{self.snapshot}.pkl"
        )

    def get(self) -> OptionChains:
        try:
            chains = self.get_from_cache()
            return chains
        except Exception as e:
            chains = self.get_from_yfinance()
            self.save_to_cache(chains)
            return chains

    def get_from_cache(self) -> OptionChains:
        if self.cache_path is None or not self.cache_path.exists():
            raise FileNotFoundError(
                "No valid cache path defined or file does not exist."
            )

        try:
            with open(self.cache_path, "rb") as f:
                curve = pickle.load(f)
                if not isinstance(curve, OptionChains):
                    raise TypeError(
                        "Cache does not contain a valid OptionChains object."
                    )
                return curve

        except Exception as e:
            raise RuntimeError(f"Failed to load option chains from cache: {e}")

    def get_from_yfinance(self) -> OptionChains:
        yahoo_client = YahooClient(self.ticker)

        spot = yahoo_client.get_last_price()
        expiries = yahoo_client.get_options_expirires()

        chains = []
        for expiry in expiries:
            calls, puts = yahoo_client.get_option_chain(expiry)
            chains.extend([calls, puts])

        return OptionChains(
            chains=pd.concat(chains, ignore_index=True), underlying_spot=spot
        )

    def save_to_cache(self, chains: OptionChains) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(chains, f)
            print(f"Option chains cached at: {self.cache_path}")
