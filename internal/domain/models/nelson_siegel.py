from typing import List, Tuple, Optional
from math import exp, sqrt

import numpy as np
from scipy.optimize import minimize


class NelsonSiegel:

    def __init__(
        self,
        b0: float,
        b1: float,
        b2: float,
        tau: float,
        tenors: Optional[List[float]] = None,
        empirical_values: Optional[List[float]] = None,
    ):
        self._b0 = b0
        self._b1 = b1
        self._b2 = b2
        self._tau = tau
        if tenors is not None and empirical_values is not None:
            self._rmse = NelsonSiegel._cost(
                tenors,
                empirical_values,
                [b0, b1, b2, tau],
            )

    def __str__(self):
        lines = [
            "Nelson-Siegel Model:",
            f"  Level (b0):     {self._b0:.6f}",
            f"  Slope (b1):     {self._b1:.6f}",
            f"  Curvature (b2): {self._b2:.6f}",
            f"  Tau:            {self._tau:.6f}",
        ]
        if self._cost is not None:
            lines.append(f"  Cost (RMSE):    {self._rmse:.6f}")
        return "\n".join(lines)

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

    def value(self, t, dt: float) -> float:
        y_t = self._value(t, self.b0, self.b1, self.b2, self.tau)
        if t + dt < 1e-6:
            return y_t
        y_t_dt = self._value(t + dt, self.b0, self.b1, self.b2, self.tau)
        return ((t + dt) * y_t_dt - t * y_t) / dt

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
    def fit(
        tenors: List[float],
        empirical_values: List[float],
        initial_guess: List[float],
        bounds: List[Tuple[float, float]],
    ) -> "NelsonSiegel":

        result = minimize(
            lambda x: NelsonSiegel._cost(tenors, empirical_values, x),
            initial_guess,
            bounds=bounds,
        )

        return NelsonSiegel(
            b0=result.x[0],
            b1=result.x[1],
            b2=result.x[2],
            tau=result.x[3],
            tenors=tenors,
            empirical_values=empirical_values,
        )
