from math import sqrt

import numpy as np


class SSVI:
    """Power-law parameterization of the SSVI volatility surface (Gatheral & Jacquier, 2013)"""

    def __init__(self, gamma: float, eta: float, rho: float):
        self._gamma = gamma
        self._eta = eta
        self._rho = rho

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def rho(self) -> float:
        return self._rho

    def phi(self, theta: float) -> float:
        return self._eta / (
            pow(theta, self._gamma) * pow((1 + theta), (1 - self._gamma))
        )

    def evaluate_total_variance(
        self, moneyness: float, year_to_maturity: float, atm_volatility: float
    ) -> float:
        theta = atm_volatility**2 * year_to_maturity
        p = self.phi(theta)
        return (
            0.5
            * theta
            * (
                1.0
                + self._rho * p * moneyness
                + sqrt((p * moneyness + self._rho) ** 2 + (1.0 - self._rho**2))
            )
        )

    def evaluate_iv(
        self, moneyness: float, year_to_maturity: float, atm_volatility: float
    ) -> float:
        total_variance = self.evaluate_total_variance(
            moneyness, year_to_maturity, atm_volatility
        )
        if total_variance < 0:
            total_variance = 0.01
        return sqrt(total_variance / year_to_maturity)

    @staticmethod
    def constraints(parameters: np.ndarray) -> bool:
        gamma, eta, rho = parameters
        if not 0 < gamma <= 0.5:
            return True
        if not abs(rho) <= 1:
            return True
        if not (2 - eta * (1 + abs(rho))) >= 0:
            return True
