from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from math import exp, sqrt

import numpy as np
from scipy.optimize import minimize


class YieldTermStructureModel(ABC):

    @abstractmethod
    def __call__(self, dt: float) -> float: ...

    @abstractmethod
    def __str__(self): ...

    @classmethod
    @abstractmethod
    def make(
        cls, tenors: List[float], yields: List[float], **kwargs
    ) -> "YieldTermStructureModel": ...

    @abstractmethod
    def value(self, dt: float) -> float: ...


class NelsonSiegel(YieldTermStructureModel):
    """Nelson-Siegel model for the yield curve term structure (Nelson & Siegel, 1987)"""

    def __init__(
        self, b0: float, b1: float, b2: float, tau: float, rmse: Optional[float] = None
    ):
        self._b0 = b0
        self._b1 = b1
        self._b2 = b2
        self._tau = tau
        self._rmse = rmse

    def __str__(self):
        lines = [
            "Nelson-Siegel Model:",
            f"  Level (b0):     {self._b0:.6f}",
            f"  Slope (b1):     {self._b1:.6f}",
            f"  Curvature (b2): {self._b2:.6f}",
            f"  Tau:            {self._tau:.6f}",
            f"  Cost (RMSE):    {self._rmse:.6f}",
        ]
        return "\n".join(lines)

    def __call__(self, dt: float) -> float:
        return self.value(dt)

    @classmethod
    def make(
        cls, tenors: List[float], empirical_values: List[float], **kwargs
    ) -> "NelsonSiegel":
        return cls._fit(tenors, empirical_values, **kwargs)

    def value(self, dt: float) -> float:
        return self._value(dt, self.b0, self.b1, self.b2, self.tau)

    @property
    def b0(self) -> float:
        return self._b0

    @property
    def b1(self) -> float:
        return self._b1

    @property
    def b2(self) -> float:
        return self._b2

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def rmse(self) -> Optional[float]:
        return self._rmse

    @staticmethod
    def _value(dt: float, b0: float, b1: float, b2: float, tau: float) -> float:
        if dt < 1e-6:
            return b0
        x = dt / tau
        return b0 + b1 * (1 - exp(-x)) / x + b2 * ((1 - exp(-x)) / x - exp(-x))

    @staticmethod
    def _cost(
        tenors: List[float], empirical_values: List[float], parameters: np.ndarray
    ) -> float:
        rmse = 0.0
        b0, b1, b2, tau = parameters
        for m, y in zip(tenors, empirical_values):
            d = y - NelsonSiegel._value(m, b0, b1, b2, tau)
            rmse += d * d
        return sqrt(rmse / len(empirical_values))

    @staticmethod
    def _fit(
        tenors: List[float],
        empirical_values: List[float],
        initial_guess: Optional[List[float]] = [0.03, -0.02, 0.02, 2.0],
        bounds: Optional[List[Tuple[float, float]]] = [
            (0.0, 0.10),
            (-0.10, 0.10),
            (-0.10, 0.10),
            (0.05, 10.0),
        ],
    ) -> "NelsonSiegel":
        result = minimize(
            lambda x: NelsonSiegel._cost(tenors, empirical_values, x),
            x0=initial_guess,
            bounds=bounds,
        )
        if not result.success:
            raise RuntimeError(f"Nelson-Siegel optimization failed: {result.message}")

        return NelsonSiegel(
            b0=result.x[0],
            b1=result.x[1],
            b2=result.x[2],
            tau=result.x[3],
            rmse=NelsonSiegel._cost(tenors, empirical_values, result.x),
        )
