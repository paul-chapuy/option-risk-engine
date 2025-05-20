from abc import ABC, abstractmethod
from contextlib import contextmanager
from math import erf, exp, log, nan, sqrt

import numpy as np
from numba import njit
from scipy.optimize import brentq

from internal.domain.assets.option import ExcerciceStyle, OptionType


@contextmanager
def temp_attr(obj, attr, new_value):
    original = getattr(obj, attr)
    setattr(obj, attr, new_value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


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
    def price(
        self,
        option_type: OptionType,
        sigma: float = None,
        q: float = None,
    ) -> float: ...

    def delta(self, option_type: OptionType, h: float = 1e-4) -> float:
        with temp_attr(self, "S", self.S + h):
            up = self.price(option_type)
        with temp_attr(self, "S", self.S - h):
            down = self.price(option_type)
        return (up - down) / (2 * h)

    def gamma(self, option_type: OptionType, h: float = 1e-4) -> float:
        with temp_attr(self, "S", self.S + h):
            up = self.price(option_type)
        with temp_attr(self, "S", self.S):
            mid = self.price(option_type)
        with temp_attr(self, "S", self.S - h):
            down = self.price(option_type)
        return (up - 2 * mid + down) / (h**2)

    def vega(self, option_type: OptionType, h: float = 1e-4) -> float:
        with temp_attr(self, "sigma", self.sigma + h):
            up = self.price(option_type)
        with temp_attr(self, "sigma", self.sigma - h):
            down = self.price(option_type)
        return (up - down) / (2 * h)

    def theta(self, option_type: OptionType, h: float = 1e-4) -> float:
        with temp_attr(self, "T", self.T - h):
            val = self.price(option_type)
        base = self.price(option_type)
        return (val - base) / h

    def rho(self, option_type: OptionType, h: float = 1e-4) -> float:
        with temp_attr(self, "r", self.r + h):
            up = self.price(option_type)
        with temp_attr(self, "r", self.r - h):
            down = self.price(option_type)
        return (up - down) / (2 * h)

    def implied_volatility(
        self,
        option_type: OptionType,
        market_price: float,
        q: float,
        initial_guess: float = None,
    ) -> float:

        self.set_dividend_yield(q)

        def f(sigma):
            self.set_volatility(sigma)
            return self.price(option_type) - market_price

        a, b = 0.01, 2.0
        if initial_guess:
            a, b = 0.5 * initial_guess, 1.5 * initial_guess

        try:
            iv = brentq(f, a, b, maxiter=20, xtol=1.0e-5, rtol=1.0e-4)
        except ValueError:  # TODO: add logs at failure
            iv = nan
        except RuntimeError:
            iv = nan
        return iv


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

    def price(
        self,
        option_type: OptionType,
        sigma: float = None,
        q: float = None,
    ) -> float:
        if sigma:
            self.set_volatility(sigma)
        if q:
            self.set_dividend_yield(q)
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
    @njit(fastmath=True)
    def _price(
        S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool
    ) -> float:

        def norm(x: float) -> float:
            return 0.5 * (1 + erf(x / sqrt(2)))

        F = S * exp((r - q) * T)
        DF = exp(-r * T)

        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        call_price = DF * (F * norm(d1) - K * norm(d2))
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

    def price(
        self,
        option_type: OptionType,
        sigma: float = None,
        q: float = None,
    ) -> float:
        if sigma:
            self.set_volatility(sigma)
        if q:
            self.set_dividend_yield(q)
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
