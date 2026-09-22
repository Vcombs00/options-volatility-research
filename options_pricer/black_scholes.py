
from dataclasses import dataclass
from math import log, sqrt, exp, pi
from scipy.stats import norm


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def price(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call", q: float = 0.0) -> float:

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = "call", q: float = 0.0) -> Greeks:

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    px = price(S, K, T, r, sigma, option_type, q)

    gamma_ = exp(-q * T) * pdf_d1 / (S * sigma * sqrt(T))
    vega_ = S * exp(-q * T) * pdf_d1 * sqrt(T)

    if option_type == "call":
        delta_ = exp(-q * T) * norm.cdf(d1)
        theta_ = (
            -S * exp(-q * T) * pdf_d1 * sigma / (2 * sqrt(T))
            - r * K * exp(-r * T) * norm.cdf(d2)
            + q * S * exp(-q * T) * norm.cdf(d1)
        )
        rho_ = K * T * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta_ = -exp(-q * T) * norm.cdf(-d1)
        theta_ = (
            -S * exp(-q * T) * pdf_d1 * sigma / (2 * sqrt(T))
            + r * K * exp(-r * T) * norm.cdf(-d2)
            - q * S * exp(-q * T) * norm.cdf(-d1)
        )
        rho_ = -K * T * exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return Greeks(price=px, delta=delta_, gamma=gamma_, vega=vega_, theta=theta_, rho=rho_)


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                 option_type: str = "call", q: float = 0.0,
                 tol: float = 1e-6, max_iter: int = 100) -> float:

    sigma = 0.3
    for _ in range(max_iter):
        try:
            px = price(S, K, T, r, sigma, option_type, q)
            vega_ = greeks(S, K, T, r, sigma, option_type, q).vega
        except ValueError:
            break
        diff = market_price - px
        if abs(diff) < tol:
            return sigma
        if vega_ < 1e-8:
            break
        sigma += diff / vega_
        if sigma <= 0:
            sigma = 0.01


    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        px = price(S, K, T, r, mid, option_type, q)
        if abs(px - market_price) < tol:
            return mid
        if px > market_price:
            hi = mid
        else:
            lo = mid
    return mid
