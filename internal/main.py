"""dummy main to test stuff"""

import os
from datetime import date
from internal.domain.assets.equity import Ticker
from internal.provider.par_curve import ParCurveProvider
from internal.provider.options_chains import OptionChainsProvider
from internal.modeling.yield_curve import PiecewiseLinear, NelsonSiegel
from internal.risk_factor.spot_curve import bootstrap_spot_curve
from internal.risk_factor.dividend_curve import build_dividend_curve
from internal.risk_factor.volatility_surface import compute_volatility_surface
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle


def plot_curve(curve, interpolated_curve, title):
    T = np.linspace(0.01, 20, 1000)
    V = [interpolated_curve.value(t) for t in T]
    plt.figure(figsize=(8, 4.5))
    plt.plot(T, V, label="fit")
    plt.scatter(curve.tenors, curve.values, label="empirical")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_vol_surface(surface, ticker):
    all_moneyness = []
    all_ttm = []
    all_iv = []

    for slice in surface.slices:
        T = slice.year_to_maturity
        for m, iv in zip(slice.moneyness_range, slice.values):
            all_moneyness.append(m)
            all_ttm.append(T)
            all_iv.append(iv)

    X = np.array(all_moneyness)
    Y = np.array(all_ttm)
    Z = np.array(all_iv)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X, Y, Z, c=Z, cmap="viridis", s=10)

    ax.set_xlabel("log(F/K)")
    ax.set_ylabel("Time to Maturity")
    ax.set_zlabel("IV")
    ax.set_title(f"Implied Volatility Surface - {ticker.value}")
    fig.colorbar(sc, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()


snapshot = "2025-05-15"
ticker = Ticker.SPY
fred_api_key = os.environ["FRED_API_KEY"]

par_curve = ParCurveProvider(snapshot, fred_api_key).get()
spot_curve = bootstrap_spot_curve(par_curve)
interpolated_spot_curve = spot_curve.calibrate(PiecewiseLinear)

option_chains = OptionChainsProvider(ticker, snapshot).get()

dividend_curve = build_dividend_curve(
    option_chains,
    interpolated_spot_curve,
    date.fromisoformat(snapshot),
    min_year_fraction=0.1,
)
with open(f"dividend_curve_{ticker.value}_{snapshot}.pkl", "wb") as f:
    pickle.dump(dividend_curve, f)

interpolated_dividend_curve = dividend_curve.calibrate(
    model=NelsonSiegel, initial_guess=[0.03, -0.02, 0.02, 0.5]
)

plot_curve(dividend_curve, interpolated_dividend_curve, "div")

vol_surface = compute_volatility_surface(
    option_chains,
    interpolated_spot_curve,
    interpolated_dividend_curve,
    date.fromisoformat(snapshot),
)
with open(f"vol_surface_{ticker.value}_{snapshot}.pkl", "wb") as f:
    pickle.dump(dividend_curve, f)

plot_vol_surface(vol_surface, ticker)
