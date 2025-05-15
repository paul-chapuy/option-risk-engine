from dataclasses import dataclass
from typing import List


@dataclass
class IVPoint:
    year_to_maturity: float
    strike: float
    moneyness: float
    forward_moneyness: float
    value: float


@dataclass
class IVSlice:
    year_to_maturity: float
    points: List[IVPoint]

    @property
    def moneyness_range(self) -> List[float]:
        return [p.forward_moneyness for p in self.points]

    @property
    def values(self) -> List[float]:
        return [p.value for p in self.points]


@dataclass
class IVSurface:
    slices: List[IVPoint]
