"""
Generates two plots:
  1. Price series with detected regime overlaid as background shading
  2. Monte Carlo simulation fan chart of forward-simulated paths

Run from the project root: python notebooks/visualize.py
Outputs PNGs into notebooks/output/
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulate import calibrate_all_assets, simulate_paths
from src.regime_detection import build_regime_features, detect_regimes, label_regimes_by_volatility

OUTPUT_DIR = "notebooks/output"


def plot_price_with_regimes(prices: pd.Series, regimes: pd.Series, asset_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices.index, prices.values, color="black", linewidth=1)

    colors = {"calm": "#a8d5a2", "stressed": "#f2a2a2", "moderate": "#f2e2a2"}
    for regime_name in regimes.unique():
        mask = regimes == regime_name
        ax.fill_between(
            regimes.index, prices.min(), prices.max(),
            where=mask, color=colors.get(regime_name, "grey"),
            alpha=0.3, label=regime_name,
        )

    ax.set_title(f"{asset_name} price with detected market regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/price_with_regimes.png", dpi=150)
    plt.close(fig)


def plot_simulation_fan(paths, asset_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(100, paths.shape[0])):
        ax.plot(paths[i], color="steelblue", alpha=0.05)

    mean_path = paths.mean(axis=0)
    ax.plot(mean_path, color="darkblue", linewidth=2, label="Mean path")

    ax.set_title(f"Monte Carlo forward simulation — {asset_name} (1000 paths)")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Simulated price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/simulation_fan_chart.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prices_df = pd.read_csv("data/sample_prices.csv", index_col="date", parse_dates=True)
    asset = prices_df.columns[0]
    prices = prices_df[asset]

    gbm_params = calibrate_all_assets(prices_df)
    mu, sigma = gbm_params.loc[asset, "mu"], gbm_params.loc[asset, "sigma"]
    paths = simulate_paths(s0=prices.iloc[-1], mu=mu, sigma=sigma, n_days=252, n_paths=1000)

    features = build_regime_features(prices)
    raw_regimes = detect_regimes(features, n_regimes=2)
    named_regimes = label_regimes_by_volatility(features, raw_regimes)

    plot_price_with_regimes(prices.loc[named_regimes.index], named_regimes, asset)
    plot_simulation_fan(paths, asset)

    print(f"Saved plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
