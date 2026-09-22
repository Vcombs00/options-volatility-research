
from dataclasses import dataclass
from math import exp, sqrt
import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    ci_95: tuple


def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0,
             n_paths: int = 200_000, antithetic: bool = True,
             seed: int | None = None) -> MCResult:

    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
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
