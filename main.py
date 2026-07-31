"""
End-to-end pipeline:
  1. Load historical price data
  2. Calibrate GBM parameters (mu, sigma) per asset from historical returns
  3. Run Monte Carlo forward simulation using calibrated parameters
  4. Detect market regimes via k-means clustering on rolling vol/momentum
  5. Allocate portfolio weights, adjusted for the current detected regime
"""

import pandas as pd

from src.simulate import calibrate_all_assets, simulate_paths, simulation_summary
from src.regime_detection import (
    build_regime_features,
    detect_regimes,
    label_regimes_by_volatility,
)
from src.portfolio import regime_adjusted_allocation, portfolio_expected_return_and_vol


def main():
    prices = pd.read_csv("data/sample_prices.csv", index_col="date", parse_dates=True)
    print(f"Loaded {len(prices)} days of price data for {len(prices.columns)} assets.\n")

    # --- Step 1: Calibrate GBM parameters per asset ---
    gbm_params = calibrate_all_assets(prices)
    print("Calibrated GBM parameters (annualised):")
    print(gbm_params.round(4), "\n")

    # --- Step 2: Forward Monte Carlo simulation for one asset ---
    asset = prices.columns[0]
    s0 = prices[asset].iloc[-1]
    mu, sigma = gbm_params.loc[asset, "mu"], gbm_params.loc[asset, "sigma"]
    paths = simulate_paths(s0=s0, mu=mu, sigma=sigma, n_days=252, n_paths=1000)
    summary = simulation_summary(paths)
    print(f"1-year Monte Carlo simulation for {asset} (1000 paths):")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")
    print()

    # --- Step 3: Regime detection on the same asset ---
    features = build_regime_features(prices[asset])
    raw_regimes = detect_regimes(features, n_regimes=2)
    named_regimes = label_regimes_by_volatility(features, raw_regimes)

    current_regime = named_regimes.iloc[-1]
    regime_counts = named_regimes.value_counts()
    print(f"Detected regimes over history:\n{regime_counts}\n")
    print(f"Most recent detected regime for {asset}: '{current_regime}'\n")

    # --- Step 4: Regime-adjusted portfolio allocation ---
    weights = regime_adjusted_allocation(gbm_params, current_regime)
    port_return, port_vol = portfolio_expected_return_and_vol(weights, gbm_params)

    print("Regime-adjusted portfolio allocation:")
    print(weights.round(4), "\n")
    print(f"Expected portfolio return: {port_return:.2%}")
    print(f"Expected portfolio volatility: {port_vol:.2%}")
    print(f"Expected Sharpe (rf=4%): {(port_return - 0.04) / port_vol:.2f}")


if __name__ == "__main__":
    main()
