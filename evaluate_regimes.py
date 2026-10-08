"""
Checks how well the k-means regime detector recovers the regime shift that
data/generate_sample_data.py injects (calm first half, stressed second half).

Run after: python data/generate_sample_data.py
    python evaluate_regimes.py
"""

import numpy as np
import pandas as pd

from src.regime_detection import (
    build_regime_features,
    detect_regimes,
    label_regimes_by_volatility,
)


def main():
    prices = pd.read_csv("data/sample_prices.csv", index_col="date", parse_dates=True)
    half = len(prices) // 2  # generator: first half calm, second half stressed

    print("Regime detection accuracy vs the known injected regime shift (synthetic data):")
    accs = []
    for asset in prices.columns:
        feats = build_regime_features(prices[asset])
        regimes = label_regimes_by_volatility(feats, detect_regimes(feats, n_regimes=2))
        positions = [prices.index.get_loc(d) for d in regimes.index]
        truth = np.array(["calm" if i < half else "stressed" for i in positions])
        acc = float((regimes.values == truth).mean())
        accs.append(acc)
        print(f"  {asset}: {acc:.1%} of {len(regimes)} days labelled correctly")
    print(f"Range: {min(accs):.1%} - {max(accs):.1%}")


if __name__ == "__main__":
    main()
