from enum import Enum, auto
from typing import List, Optional
from dataclasses import dataclass

from internal.modeling.yield_term_structure import YieldTermStructureModel


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


class TermStructureType(Enum):
    YTM = auto()
    Spot = auto()
    Divivend = auto()


@dataclass
class YieldPoint:
    year_to_maturity: float
    yield_rate: float


@dataclass
class YieldTermStructure:

    points: List[YieldPoint]
    term_stucture_type: TermStructureType
    model: Optional[YieldTermStructureModel] = None

    def __post_init__(self):
        self.points.sort(key=lambda p: p.year_to_maturity)

    def __iter__(self):
        return ((p.year_to_maturity, p.yield_rate) for p in self.points)

    def tenors(self) -> List[float]:
        return [p.year_to_maturity for p in self.points]

    def yields(self) -> List[float]:
        return [p.yield_rate for p in self.points]

    @classmethod
    def make(
        cls,
        points: List[YieldPoint],
        term_structure_type: TermStructureType,
        model: YieldTermStructureModel,
    ) -> "YieldTermStructure":
        ts = cls(points, term_structure_type)
        ts.model = model.make(ts.tenors(), ts.yields())
        return ts
