"""
Black-Scholes-Merton pricer and Greeks, implemented from the closed-form
formulas (not wrapping an existing pricing library). Supports a continuous
dividend yield q, so it also covers stocks with dividends, FX (q = foreign
rate), and futures options (q = r) as special cases.

Notation:
    S     spot price of the underlying
    K     strike price
    T     time to expiry, in years
    r     risk-free rate (continuously compounded)
    sigma implied/realized volatility (annualized)
    q     continuous dividend yield (default 0)
"""

from dataclasses import dataclass
from math import log, sqrt, exp, pi
from scipy.stats import norm


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float   # per 1.00 change in vol (i.e. per 100 vol points); divide by 100 for "per 1%"
    theta: float  # per year; divide by 365 for per-day
    rho: float    # per 1.00 change in rate; divide by 100 for "per 1%"


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def price(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = "call", q: float = 0.0) -> Greeks:
    """
    Full set of first-order (and gamma, second-order) Greeks in closed form.

    delta: sensitivity to spot
    gamma: sensitivity of delta to spot (same for calls and puts)
    vega:  sensitivity to vol (same for calls and puts)
    theta: sensitivity to time decay (raw, per year -- see note above)
    rho:   sensitivity to interest rate
    """
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
    """
    Back out implied volatility from a market price using Newton-Raphson,
    falling back to bisection if Newton doesn't converge (vega can be tiny
    for deep ITM/OTM or very short-dated options, which makes Newton unstable).
    """
    # Newton-Raphson using vega as the derivative
    sigma = 0.3  # reasonable starting guess
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

    # Bisection fallback -- robust even when Newton fails
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
