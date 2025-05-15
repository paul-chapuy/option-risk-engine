from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum, auto
from math import log
from datetime import date

from internal.domain.assets.equity import Equity


class OptionType(Enum):
    Call = auto()
    Put = auto()


class ExcerciceStyle(Enum):
    European = auto()
    American = auto()


@dataclass
class Option:
    underlying: Equity
    strike: float
    expiry: date
    option_type: OptionType
    excercice_style: ExcerciceStyle
    symbol: Optional[str] = None
    last_trade_date: Optional[date] = None
    last_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    in_the_money: Optional[bool] = None


@dataclass
class OptionChain:
    underlying: Equity
    expiry: date
    options: List[Option]

    def atm_pair(self) -> Optional[Tuple[float, Option, Option]]:
        best_pair = None
        closest_moneyness = float("inf")
        strike_to_options: Dict[float, Dict[OptionType, Option]] = {}

        for option in self.options:
            if option.strike <= 0:
                continue
            strike_to_options.setdefault(option.strike, {})[option.option_type] = option

        spot = self.underlying.last_close_price
        if spot <= 0:
            return None

        for strike, ops in strike_to_options.items():
            call = ops.get(OptionType.Call)
            put = ops.get(OptionType.Put)

            if call and put:
                try:
                    moneyness = log(spot / strike)
                except ValueError:
                    continue

                if abs(moneyness) < closest_moneyness:
                    closest_moneyness = abs(moneyness)
                    best_pair = (strike, call, put)

        return best_pair


@dataclass
class OptionChains:
    underlying: Equity
    exercice_style: ExcerciceStyle
    chains: List[OptionChain]
