from enum import Enum, auto
from typing import List, Optional
from datetime import date
from dataclasses import dataclass

from internal.modeling.yield_curve import YieldCurveModel


class YieldCurveType(Enum):
    Par = auto()
    Spot = auto()
    Divivend = auto()


@dataclass
class YieldPoint:
    year_to_maturity: float
    yield_rate: float
    evaluation_date: date


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
        return ((p.year_to_maturity, p.yield_rate) for p in self.points)

    @property
    def tenors(self) -> List[float]:
        return [p.year_to_maturity for p in self.points]

    @property
    def yields(self) -> List[float]:
        return [p.yield_rate for p in self.points]

    @staticmethod
    def calibrate(
        curve: "YieldCurve", model: YieldCurveModel, **kwargs
    ) -> "YieldCurveModel":
        return model.make(curve.tenors, curve.yields, **kwargs)
