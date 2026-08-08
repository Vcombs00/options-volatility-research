"""
Monte Carlo pricer for European options under geometric Brownian motion (GBM).

Purpose: a third, independent pricing method to cross-check Black-Scholes and
the binomial tree against. It's also the natural base to extend later toward
payoffs that don't have closed forms (Asian options, barrier options,
basket/spread options) -- anything path-dependent.

Uses antithetic variates for variance reduction: for every random draw z,
also simulate the mirrored draw -z. This cuts simulation noise substantially
for roughly the same compute, since it cancels out first-order sampling
error without introducing any bias.
"""

from dataclasses import dataclass
from math import exp, sqrt
import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float          # standard error of the price estimate
    ci_95: tuple               # (low, high) 95% confidence interval


def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0,
             n_paths: int = 200_000, antithetic: bool = True,
             seed: int | None = None) -> MCResult:
    """
    Price a European option via Monte Carlo simulation of terminal GBM prices.

    Since only the terminal value matters for a European payoff, we simulate
    S_T directly from the closed-form GBM solution rather than stepping
    through a full price path -- this is exact (no discretization error)
    and much faster than a time-stepped simulation.

        S_T = S * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])  # antithetic pairs
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * sqrt(T) * z
    S_T = S * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = exp(-r * T) * payoffs
    price_est = discounted.mean()
    std_error = discounted.std(ddof=1) / sqrt(len(discounted))
    ci_95 = (price_est - 1.96 * std_error, price_est + 1.96 * std_error)

    return MCResult(price=price_est, std_error=std_error, ci_95=ci_95)
