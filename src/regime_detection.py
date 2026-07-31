"""
Market regime detection via k-means clustering.

Rather than clustering on raw prices (which are non-stationary and
carry no direct regime signal), each day is described by a small set
of rolling statistical features - realised volatility and momentum -
computed over a trailing window. K-means then partitions days into
regimes (e.g. 'calm/trending' vs 'stressed/volatile') based on these
features.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_regime_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Build a feature set for regime detection from a single asset's
    price series:
      - rolling realised volatility (annualised)
      - rolling momentum (cumulative return over the window)
    """
    log_returns = np.log(prices / prices.shift(1))
    rolling_vol = log_returns.rolling(window).std() * np.sqrt(252)
    rolling_momentum = prices.pct_change(window)

    features = pd.DataFrame(
        {"volatility": rolling_vol, "momentum": rolling_momentum}
    ).dropna()
    return features


def detect_regimes(features: pd.DataFrame, n_regimes: int = 2, random_state: int = 42) -> pd.Series:
    """
    Cluster days into n_regimes market regimes using k-means on
    standardised volatility/momentum features.

    Returns
    -------
    pd.Series indexed like `features`, with integer cluster labels.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    labels = km.fit_predict(scaled)

    return pd.Series(labels, index=features.index, name="regime")


def label_regimes_by_volatility(features: pd.DataFrame, regimes: pd.Series) -> pd.Series:
    """
    Relabel raw cluster indices as human-readable regime names, ranked
    by mean volatility within each cluster (lowest -> 'calm', highest
    -> 'stressed', with intermediate labels if more than two clusters).
    """
    vol_by_cluster = features.groupby(regimes)["volatility"].mean().sort_values()
    ordered_clusters = vol_by_cluster.index.tolist()
    n = len(ordered_clusters)

    if n == 2:
        names = ["calm", "stressed"]
    elif n == 3:
        names = ["calm", "moderate", "stressed"]
    else:
        names = [f"regime_{i}" for i in range(n)]

    label_map = {cluster: names[i] for i, cluster in enumerate(ordered_clusters)}

    return regimes.map(label_map)
