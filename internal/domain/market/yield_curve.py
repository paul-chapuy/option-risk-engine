from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import List, Optional

from internal.modeling.yield_curve import YieldCurveModel


class YieldCurveType(Enum):
    Par = auto()
    Spot = auto()
    Dividend = auto()


@dataclass
class YieldPoint:
    year_to_maturity: float
    value: float
    evaluation_date: Optional[date] = None


@dataclass
class YieldCurve:

    points: List[YieldPoint]
    yield_type: YieldCurveType
    evaluation_date: Optional[date] = None

    def __post_init__(self):
        self.points.sort(key=lambda p: p.year_to_maturity)
        unique_dates = set(p.evaluation_date for p in self.points)
        if len(unique_dates) != 1:
            raise ValueError("All yield points must have the same evaluation date.")
        self.evaluation_date = unique_dates.pop()

    def __iter__(self):
        return ((p.year_to_maturity, p.value) for p in self.points)

    @property
    def tenors(self) -> List[float]:
        return [p.year_to_maturity for p in self.points]

    @property
    def values(self) -> List[float]:
        return [p.value for p in self.points]

    def calibrate(self, model: YieldCurveModel, **kwargs) -> "YieldCurveModel":
        return model.make(self.tenors, self.values, **kwargs)
