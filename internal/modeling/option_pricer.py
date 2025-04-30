from math import exp, sqrt, log, erf
from abc import ABC, abstractmethod

from internal.domain.assets.option import OptionType, ExcerciceStyle

import numpy as np
from numba import njit


class OptionPricer(ABC):
    """Abstract base class for vanilla option pricers (European or American)."""

    def __init__(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = 0.0
        self.q = 0.0

    def make(
        exercise_style: ExcerciceStyle,
        S: float,
        K: float,
        r: float,
        T: float,
    ) -> "OptionPricer":
        match exercise_style:
            case ExcerciceStyle.European:
                return BlackScholesMerton(S, K, r, T)
            case ExcerciceStyle.American:
                return CoxRossRubinstein(S, K, r, T)
            case _:
                raise ValueError(f"Unsupported exercise style: {exercise_style}")

    def set_volatility(self, sigma: float) -> None:
        self.sigma = sigma

    def set_dividend_yield(self, q: float) -> None:
        self.q = q

    @abstractmethod
    def price(self, option_type: OptionType) -> float: ...


class BlackScholesMerton(OptionPricer):
    """Closed-form pricing model for European options (Black-Scholes-Merton, 1973)."""

    def __init__(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
    ):
        super().__init__(S, K, r, T)

    def price(self, option_type: OptionType) -> float:
        if self.T <= 0:
            return (
                max(self.S - self.K, 0.0)
                if option_type == OptionType.Call
                else max(self.K - self.S, 0.0)
            )

        S, K, r, q, sigma, T = self.S, self.K, self.r, self.q, self.sigma, self.T
        if option_type == OptionType.Call:
            return BlackScholesMerton._price(S, K, r, q, sigma, T, is_call=True)
        return BlackScholesMerton._price(S, K, r, q, sigma, T, is_call=False)

    @staticmethod
    @njit
    def _price(
        S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool
    ) -> float:
        F = S * exp((r - q) * T)
        DF = exp(-r * T)

        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
        Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)))
        call_price = DF * (F * Nd1 - K * Nd2)
        if is_call:
            return call_price

        return call_price - DF * (F - K)


class CoxRossRubinstein(OptionPricer):
    """Binomial tree model for American options (Cox, Ross & Rubinstein, 1979)."""

    def __init__(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
    ):
        super().__init__(S, K, r, T)

    def price(self, option_type: OptionType) -> float:
        if self.T <= 0:
            return (
                max(self.S - self.K, 0.0)
                if option_type == OptionType.Call
                else max(self.K - self.S, 0.0)
            )

        S, K, r, q, sigma, T = self.S, self.K, self.r, self.q, self.sigma, self.T
        if option_type == OptionType.Call:
            return CoxRossRubinstein._price(S, K, r, q, sigma, T, is_call=True)
        return CoxRossRubinstein._price(S, K, r, q, sigma, T, is_call=False)

    @staticmethod
    @njit(fastmath=True)
    def _price(
        S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool
    ) -> float:
        N = 1000
        dt = T / N

        u = exp(sigma * sqrt(dt))
        d = 1 / u
        p = (exp((r - q) * dt) - d) / (u - d)
        df = exp(-r * dt)

        ST = np.empty(N + 1)
        for j in range(N + 1):
            ST[j] = S * u ** (N - j) * d**j

        payoff = np.empty(N + 1)
        for j in range(N + 1):
            payoff[j] = max(ST[j] - K, 0.0) if is_call else max(K - ST[j], 0.0)

        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                ST[j] = ST[j] / u
                cont_val = df * (p * payoff[j] + (1 - p) * payoff[j + 1])
                exercise_val = max(ST[j] - K, 0.0) if is_call else max(K - ST[j], 0.0)
                payoff[j] = max(cont_val, exercise_val)

        return payoff[0]
