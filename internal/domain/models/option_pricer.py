from math import exp, sqrt, log
from abc import ABC, abstractmethod

from internal.domain.assets.option import OptionType, ExcerciceStyle

from scipy.stats import norm


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

    @property
    def DF(self) -> float:
        return exp(-self.r * self.T)

    @property
    def F(self) -> float:
        return self.S * exp((self.r - self.q) * self.T)

    def set_volatility(self, volatility: float) -> None:
        self.volatility = volatility

    def set_dividend_yield(self, dividend_yield: float) -> None:
        self.dividend_yield = dividend_yield

    @abstractmethod
    def price(self, option_type: OptionType) -> float:
        pass


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

    @property
    def d1(self) -> float:
        return (self.moneyness + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * sqrt(self.T)
        )

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * sqrt(self.T)

    def price(self, option_type: OptionType) -> float:
        if self.T <= 0:
            return (
                max(self.S - self.K, 0)
                if option_type == OptionType.CALL
                else max(self.K - self.S, 0)
            )

        call_price = self.DF * (self.F * norm.cdf(self.d1) - self.K * norm.cdf(self.d2))

        if option_type == OptionType.CALL:
            return call_price

        return call_price - self.DF * (self.F - self.K)


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
        return NotImplementedError("CRR not implemented yet.")
