"""
Cox-Ross-Rubinstein binomial tree pricer.

Why this exists alongside Black-Scholes: BS only prices European options
(exercise at expiry only). Most listed equity options are American (early
exercise allowed), and the early-exercise premium can matter, especially for
puts and for calls on dividend-paying stocks near an ex-div date. A binomial
tree handles both European and American exercise and converges to the BS
price as steps -> infinity, which is also a useful way to sanity-check your
BS implementation.
"""

from math import exp, sqrt


def binomial_price(S: float, K: float, T: float, r: float, sigma: float,
                    N: int = 200, option_type: str = "call",
                    american: bool = True, q: float = 0.0) -> float:
    """
    Price a European or American option via a CRR binomial tree.

    N: number of time steps. Larger N = more accurate, slower. 200-500 is
       plenty for convergence to ~1e-3 for typical parameters.
    """
    if T <= 0 or sigma <= 0 or N < 1:
        raise ValueError("T, sigma must be positive and N >= 1")

    dt = T / N
    u = exp(sigma * sqrt(dt))          # up factor
    d = 1 / u                          # down factor
    disc = exp(-r * dt)
    p = (exp((r - q) * dt) - d) / (u - d)  # risk-neutral up probability

    if not (0 < p < 1):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} out of (0,1); "
            "try increasing N or check inputs (sigma, r, dt)."
        )

    # Terminal payoffs at maturity (bottom-up: j = number of down moves)
    terminal_prices = [S * (u ** (N - j)) * (d ** j) for j in range(N + 1)]
    if option_type == "call":
        values = [max(sp - K, 0.0) for sp in terminal_prices]
    elif option_type == "put":
        values = [max(K - sp, 0.0) for sp in terminal_prices]
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Backward induction through the tree
    for step in range(N - 1, -1, -1):
        new_values = []
        for j in range(step + 1):
            continuation = disc * (p * values[j] + (1 - p) * values[j + 1])
            if american:
                spot_at_node = S * (u ** (step - j)) * (d ** j)
                if option_type == "call":
                    exercise = max(spot_at_node - K, 0.0)
                else:
                    exercise = max(K - spot_at_node, 0.0)
                new_values.append(max(continuation, exercise))
            else:
                new_values.append(continuation)
        values = new_values

    return values[0]
