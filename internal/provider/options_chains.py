import pickle
from pathlib import Path
from typing import Final, Dict
from datetime import date

from internal.domain.assets.option import (
    Option,
    OptionChain,
    OptionChains,
    OptionType,
    ExcerciceStyle,
)
from internal.domain.assets.equity import Equity, Ticker
from internal.infra.api.yahoo_finance import YahooClient


class OptionChainsProvider:

    TICKER_TO_EXERCICE_STYLE: Dict[Ticker, ExcerciceStyle] = {
        Ticker.SPY: ExcerciceStyle.American,
        Ticker.AAPL: ExcerciceStyle.American,
        Ticker.AMZN: ExcerciceStyle.American,
        Ticker.TSLA: ExcerciceStyle.American,
        Ticker.MSFT: ExcerciceStyle.American,
        Ticker.SPX: ExcerciceStyle.European,
    }

    def __init__(self, ticker: Ticker, snapshot: str):
        self.ticker = ticker
        self.snapshot = snapshot
        self.cache_path = (
            Path(__file__).resolve().parents[2]
            / f"cache/option_chains_{self.ticker}_{self.snapshot}.pkl"
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

        with open(self.cache_path, "rb") as f:
            curve = pickle.load(f)
            if not isinstance(curve, OptionChains):
                raise TypeError("Cache does not contain a valid OptionChains object.")
            return curve

    def get_from_yfinance(self) -> OptionChains:
        yahoo_client = YahooClient(self.ticker.value)

        equity = Equity(
            ticker=self.ticker,
            last_price=yahoo_client.get_last_price(),
            last_close_price=yahoo_client.get_last_close(),
        )

        chains = []
        expiries = yahoo_client.get_options_expirires()
        for expiry in expiries:
            calls, puts = yahoo_client.get_option_chain(expiry)

            def parse_option_row(row, option_type: OptionType) -> Option:
                return Option(
                    underlying=equity,
                    strike=row.strike,
                    expiry=date.fromisoformat(expiry),
                    option_type=option_type,
                    excercice_style=ExcerciceStyle.American,
                    symbol=row.contractSymbol,
                    last_trade_date=row.lastTradeDate,
                    last_price=row.lastPrice,
                    bid=row.bid,
                    ask=row.ask,
                    volume=row.volume,
                    open_interest=row.openInterest,
                    in_the_money=row.inTheMoney,
                )

            options = []
            for row in calls.itertuples(index=False):
                options.append(parse_option_row(row, OptionType.Call))
            for row in puts.itertuples(index=False):
                options.append(parse_option_row(row, OptionType.Put))

            chains.append(
                OptionChain(
                    options=options,
                    expiry=date.fromisoformat(expiry),
                    underlying=equity,
                )
            )

        return OptionChains(
            chains=chains,
            underlying=equity,
            exercice_style=self.TICKER_TO_EXERCICE_STYLE[equity.ticker],
        )

    def save_to_cache(self, chains: OptionChains) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(chains, f)
            print(f"Option chains cached at: {self.cache_path}")
