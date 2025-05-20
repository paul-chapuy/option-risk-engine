from abc import ABC, abstractmethod
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np


class VolatilitySurfaceModel(ABC):

    @classmethod
    @abstractmethod
    def make(
        cls,
        fwd_moneyness_ranges: List[List[float]],
        values: List[List[float]],
        **kwargs,
    ) -> "VolatilitySurfaceModel": ...

    @abstractmethod
    def value(self, **kwargs) -> float: ...


class SSVI(VolatilitySurfaceModel):
    """Power-law parameterization of SSVI vol surface (Gatheral & Jacquier, 2013)"""

    def __init__(
        self, gamma: float, eta: float, rho: float, rmse: Optional[float] = None
    ):
        self._gamma = gamma
        self._eta = eta
        self._rho = rho
        self._rmse = rmse

    @classmethod
    def make(
        cls,
        fwd_moneyness_ranges: List[List[float]],
        values: List[List[float]],
        vegas: List[List[float]],
        **kwargs,
    ) -> "SSVI":
        return cls._fit(fwd_moneyness_ranges, values, vegas, **kwargs)

    def __str__(self):
        lines = [
            "SSVI Model:",
            f"  Gamma:      {self._gamma:.6f}",
            f"  Eta:        {self._eta:.6f}",
            f"  Rho:        {self._rho:.6f}",
            f"  Cost (RMSE):{self._rmse:.6f}",
        ]
        return "\n".join(lines)

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def rho(self) -> float:
        return self._rho

    def _phi(self, theta: float) -> float:
        return self._eta / (
            pow(theta, self._gamma) * pow((1 + theta), (1 - self._gamma))
        )

    def _value_total_variance(
        self, moneyness: float, year_to_maturity: float, atm_volatility: float
    ) -> float:
        theta = atm_volatility**2 * year_to_maturity
        p = self._phi(theta)
        return (
            0.5
            * theta
            * (
                1.0
                + self._rho * p * moneyness
                + sqrt((p * moneyness + self._rho) ** 2 + (1.0 - self._rho**2))
            )
        )

    def value(
        self, moneyness: float, year_to_maturity: float, atm_volatility: float
    ) -> float:
        total_variance = self._value_total_variance(
            moneyness, year_to_maturity, atm_volatility
        )
        if total_variance < 0:
            total_variance = 0.01
        return sqrt(total_variance / year_to_maturity)

    @staticmethod
    def _cost(
        fwd_moneyness_ranges: List[List[float]],
        values: List[List[float]],
        vegas: List[List[float]],
        parameters: np.ndarray,
    ) -> float:
        raise NotImplementedError()

    @staticmethod
    def _fit(
        fwd_moneyness_ranges: List[List[float]],
        values: List[List[float]],
        vegas: List[List[float]],
        initial_guess: Optional[List[float]],
        bounds: Optional[List[Tuple[float, float]]],
    ) -> "SSVI":
        raise NotImplementedError()

    @staticmethod
    def _constraints(parameters: np.ndarray) -> bool:
        gamma, eta, rho = parameters
        if not 0 < gamma <= 0.5:
            return True
        if not abs(rho) <= 1:
            return True
        if not (2 - eta * (1 + abs(rho))) >= 0:
            return True
