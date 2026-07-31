"""
Generates synthetic daily price data for a small basket of assets.

In a real setting this would be replaced with actual historical prices
(e.g. pulled from a market data provider). For this project, synthetic
data lets the simulation and regime-detection pipeline be fully
reproducible and self-contained.

Each asset's price path is generated as Geometric Brownian Motion with
a randomly chosen drift/volatility regime that shifts partway through
the series, so the regime-detection step has real structure to find.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_DAYS = 756  # ~3 trading years
ASSETS = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"]
START_PRICE = 100.0
DT = 1 / 252  # one trading day, in years


def simulate_gbm_path(n_days, mu, sigma, s0=START_PRICE, dt=DT):
    """Simulate a single Geometric Brownian Motion price path."""
    prices = np.zeros(n_days)
    prices[0] = s0
    z = np.random.standard_normal(n_days - 1)
    for t in range(1, n_days):
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[t - 1]
        )
    return prices


def simulate_regime_switching_path(n_days):
    """
    Simulate a price path with a low-volatility 'calm' regime for the
    first half and a high-volatility 'stressed' regime for the second
    half, to give the k-means regime detector real signal to find.
    """
    half = n_days // 2
    calm = simulate_gbm_path(half, mu=0.08, sigma=0.12)
    stressed = simulate_gbm_path(n_days - half, mu=-0.02, sigma=0.35, s0=calm[-1])
    return np.concatenate([calm, stressed])


def main():
    dates = pd.bdate_range(start="2023-01-02", periods=N_DAYS)
    data = {asset: simulate_regime_switching_path(N_DAYS) for asset in ASSETS}
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    df.to_csv("data/sample_prices.csv")
    print(f"Wrote data/sample_prices.csv with {N_DAYS} rows, {len(ASSETS)} assets.")


if __name__ == "__main__":
    main()
