from dataclasses import dataclass
from pandas import DataFrame


@dataclass
class OptionChains:
    chains: DataFrame
    underlying_spot: float
