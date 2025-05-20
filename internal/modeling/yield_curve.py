from abc import ABC, abstractmethod
from bisect import bisect_right
from math import exp, sqrt
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


class YieldCurveModel(ABC):

    @classmethod
    @abstractmethod
    def make(
        cls, tenors: List[float], yields: List[float], **kwargs
    ) -> "YieldCurveModel": ...

    @abstractmethod
    def value(self, dt: float) -> float: ...


class NelsonSiegel(YieldCurveModel):
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


class PiecewiseLinear(YieldCurveModel):
    """Piecewise linear interpolation for yield curve"""

    def __init__(self, tenors: List[float], yields: List[float]):
        if len(tenors) != len(yields):
            raise ValueError("Tenors and yields must have the same length.")
        if not all(earlier <= later for earlier, later in zip(tenors, tenors[1:])):
            raise ValueError("Tenors must be sorted in increasing order.")
        self._tenors = tenors
        self._yields = yields

    def value(self, dt: float) -> float:
        if dt <= self._tenors[0]:
            return self._yields[0]
        elif dt >= self._tenors[-1]:
            return self._yields[-1]
        else:
            i = bisect_right(self._tenors, dt)
            t0, t1 = self._tenors[i - 1], self._tenors[i]
            y0, y1 = self._yields[i - 1], self._yields[i]
            weight = (dt - t0) / (t1 - t0)
            return y0 + weight * (y1 - y0)

    @classmethod
    def make(
        cls, tenors: List[float], yields: List[float], **kwargs
    ) -> "PiecewiseLinear":
        return cls(tenors, yields)
