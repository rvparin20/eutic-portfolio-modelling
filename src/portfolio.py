"""
Simple regime-aware portfolio allocation.

Given calibrated per-asset GBM parameters and a detected market
regime, this module allocates portfolio weights using a basic
Sharpe-maximising heuristic within regime, tilting toward higher
mu/sigma assets in calm regimes and toward capital preservation
(lower volatility assets) in stressed regimes.
"""

import numpy as np
import pandas as pd


def sharpe_weighted_allocation(gbm_params: pd.DataFrame, risk_free_rate: float = 0.04) -> pd.Series:
    """
    Allocate weights proportional to each asset's estimated Sharpe
    ratio (clipped at zero, since negative-Sharpe assets get no
    allocation in this simple heuristic).
    """
    sharpe = (gbm_params["mu"] - risk_free_rate) / gbm_params["sigma"]
    sharpe_clipped = sharpe.clip(lower=0)

    if sharpe_clipped.sum() == 0:
        # No asset has positive Sharpe: fall back to equal weight
        n = len(gbm_params)
        return pd.Series(1 / n, index=gbm_params.index)

    weights = sharpe_clipped / sharpe_clipped.sum()
    return weights


def regime_adjusted_allocation(
    gbm_params: pd.DataFrame, current_regime: str, stressed_cash_buffer: float = 0.30
) -> pd.Series:
    """
    Adjust the Sharpe-weighted allocation based on the current
    detected regime. In a 'stressed' regime, a portion of the
    portfolio is held back as an implicit cash buffer by scaling down
    risky-asset weights.
    """
    base_weights = sharpe_weighted_allocation(gbm_params)

    if current_regime == "stressed":
        scaled = base_weights * (1 - stressed_cash_buffer)
        return scaled
    return base_weights


def portfolio_expected_return_and_vol(
    weights: pd.Series, gbm_params: pd.DataFrame, corr_matrix: pd.DataFrame = None
) -> tuple[float, float]:
    """
    Compute expected portfolio return and volatility given asset
    weights and calibrated GBM parameters. Assumes zero correlation
    between assets if no correlation matrix is supplied (a
    simplifying assumption, noted as such).
    """
    mu = gbm_params["mu"]
    sigma = gbm_params["sigma"]

    port_return = float((weights * mu).sum())

    if corr_matrix is None:
        # Zero-correlation simplifying assumption
        port_var = float(((weights * sigma) ** 2).sum())
    else:
        cov = np.outer(sigma, sigma) * corr_matrix.values
        port_var = float(weights.values @ cov @ weights.values)

    return port_return, np.sqrt(port_var)
