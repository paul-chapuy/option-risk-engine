from typing import List, Optional, Tuple
from dataclasses import dataclass

from internal.modeling.yield_term_structure import YieldTermStructureModel


@dataclass
class YieldPoint:
    maturity: float
    yield_rate: float


@dataclass
class YieldTermStructure:

    points: List[YieldPoint]
    model: Optional[YieldTermStructureModel] = None

    def tenors(self) -> List[float]:
        return [p.maturity for p in self.points]

    def yields(self) -> List[float]:
        return [p.yield_rate for p in self.points]

    def fit(
        self,
        model: YieldTermStructureModel,
    ) -> None:
        tenors = self.tenors()
        rates = self.yields()
        self.model = model.make(tenors, rates)
