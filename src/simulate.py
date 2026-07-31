"""
Stochastic asset-path simulation with parameter calibration.

Models each asset as Geometric Brownian Motion (GBM):

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

Parameters (mu, sigma) are calibrated from historical daily log
returns via maximum likelihood estimation, which for GBM reduces to
simple sample mean/variance of log returns, annualised.
"""

import numpy as np
import pandas as pd


def calibrate_gbm_params(prices: pd.Series, trading_days: int = 252) -> tuple[float, float]:
    """
    Estimate annualised drift (mu) and volatility (sigma) for a GBM
    process from a historical price series.

    Returns
    -------
    (mu, sigma) : tuple of float
        Annualised drift and volatility estimates.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_mean = log_returns.mean()
    daily_var = log_returns.var()

    # Annualise: for GBM, E[log return] = (mu - 0.5*sigma^2) * dt
    sigma = np.sqrt(daily_var * trading_days)
    mu = daily_mean * trading_days + 0.5 * sigma**2
    return mu, sigma


def calibrate_all_assets(price_df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate GBM parameters for every asset column in a price DataFrame."""
    records = []
    for asset in price_df.columns:
        mu, sigma = calibrate_gbm_params(price_df[asset])
        records.append({"asset": asset, "mu": mu, "sigma": sigma})
    return pd.DataFrame(records).set_index("asset")


def simulate_paths(
    s0: float, mu: float, sigma: float, n_days: int, n_paths: int, dt: float = 1 / 252
) -> np.ndarray:
    """
    Monte Carlo simulation of forward GBM price paths using calibrated
    parameters.

    Returns
    -------
    np.ndarray of shape (n_paths, n_days)
    """
    z = np.random.standard_normal((n_paths, n_days - 1))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return s0 * np.exp(log_paths)


def simulation_summary(paths: np.ndarray) -> dict:
    """Summary statistics (mean, 5th/95th percentile final price) for simulated paths."""
    final_prices = paths[:, -1]
    return {
        "mean_final_price": float(np.mean(final_prices)),
        "p05_final_price": float(np.percentile(final_prices, 5)),
        "p95_final_price": float(np.percentile(final_prices, 95)),
        "prob_below_start": float(np.mean(final_prices < paths[0, 0])),
    }
