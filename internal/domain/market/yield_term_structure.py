from enum import Enum, auto
from typing import List, Optional
from datetime import date
from dataclasses import dataclass

from internal.modeling.yield_term_structure import YieldTermStructureModel


class TermStructureType(Enum):
    Par = auto()
    Spot = auto()
    Divivend = auto()


@dataclass
class YieldPoint:
    year_to_maturity: float
    yield_rate: float
    evaluation_date: date


@dataclass
class YieldTermStructure:

    points: List[YieldPoint]
    term_stucture_type: TermStructureType
    evaluation_date: Optional[date] = None
    model: Optional[YieldTermStructureModel] = None

    def __post_init__(self):
        self.points.sort(key=lambda p: p.year_to_maturity)
        unique_dates = set(p.evaluation_date for p in self.points)
        if len(unique_dates) != 1:
            raise ValueError("All yield points must have the same evaluation date.")
        self.evaluation_date = unique_dates.pop()

    def __iter__(self):
        return ((p.year_to_maturity, p.yield_rate) for p in self.points)

    def tenors(self) -> List[float]:
        return [p.year_to_maturity for p in self.points]

    def yields(self) -> List[float]:
        return [p.yield_rate for p in self.points]

    @staticmethod
    def from_raw_data(
        points: List[YieldPoint],
        term_structure_type: TermStructureType,
        model: YieldTermStructureModel,
    ) -> "YieldTermStructure":
        ts = YieldTermStructure(points, term_structure_type)
        ts.model = model.make(ts.tenors(), ts.yields())
        return ts

    @staticmethod
    def from_fitted_model(
        points: List[YieldPoint],
        term_structure_type: TermStructureType,
        fitted_model: YieldTermStructureModel,
    ) -> "YieldTermStructure":
        ts = YieldTermStructure(points, term_structure_type)
        ts.model = fitted_model
        return ts
