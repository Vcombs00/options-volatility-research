"""
Demo / validation script.

Run this to:
  1. Price a sample option and print price + Greeks.
  2. Cross-check the Black-Scholes price against the binomial tree
     (they should converge as N grows -- this is your unit test that
     both implementations are internally consistent).
  3. Plot delta, gamma, vega, and theta as functions of spot price, so you
     can visually confirm the shapes match known option theory:
       - delta: S-shaped, 0 -> 1 for calls as spot rises
       - gamma: peaks near the strike (ATM), decays away from it
       - vega: peaks near the strike, decays away from it
       - theta: typically negative, steepest decay near the strike
  4. Back out implied vol from a price to confirm implied_vol() inverts price().
"""

import numpy as np
import matplotlib.pyplot as plt

from options_pricer import price, greeks, binomial_price, implied_vol

# ---- Sample option parameters ----
S = 100.0     # spot
K = 100.0     # strike (at-the-money)
T = 0.5       # 6 months to expiry
r = 0.05      # 5% risk-free rate
sigma = 0.20  # 20% annualized vol
q = 0.0       # no dividend
option_type = "call"

# ---- 1. Basic price + Greeks ----
g = greeks(S, K, T, r, sigma, option_type, q)
print("=== Black-Scholes ===")
print(f"Price: {g.price:.4f}")
print(f"Delta: {g.delta:.4f}")
print(f"Gamma: {g.gamma:.4f}")
print(f"Vega (per 1.00 vol): {g.vega:.4f}  (per 1% vol: {g.vega / 100:.4f})")
print(f"Theta (per year): {g.theta:.4f}  (per day: {g.theta / 365:.4f})")
print(f"Rho (per 1.00 rate): {g.rho:.4f}  (per 1% rate: {g.rho / 100:.4f})")

# ---- 2. Cross-check against binomial tree ----
bs_price = price(S, K, T, r, sigma, option_type, q)
bin_price_euro = binomial_price(S, K, T, r, sigma, N=500, option_type=option_type, american=False, q=q)
bin_price_amer = binomial_price(S, K, T, r, sigma, N=500, option_type=option_type, american=True, q=q)
print("\n=== Cross-check ===")
print(f"BS price:                 {bs_price:.4f}")
print(f"Binomial (European, N=500): {bin_price_euro:.4f}  (should closely match BS)")
print(f"Binomial (American, N=500): {bin_price_amer:.4f}  (>= European price; equal for non-dividend calls)")

# ---- 3. Plot Greeks vs spot price ----
spots = np.linspace(60, 140, 200)
deltas, gammas, vegas, thetas = [], [], [], []
for s in spots:
    gr = greeks(s, K, T, r, sigma, option_type, q)
    deltas.append(gr.delta)
    gammas.append(gr.gamma)
    vegas.append(gr.vega)
    thetas.append(gr.theta)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(f"Greeks vs Spot Price (K={K}, T={T}y, sigma={sigma}, r={r}) -- {option_type}")

axes[0, 0].plot(spots, deltas, color="tab:blue")
axes[0, 0].axvline(K, color="gray", linestyle="--", linewidth=1)
axes[0, 0].set_title("Delta")
axes[0, 0].set_xlabel("Spot")

axes[0, 1].plot(spots, gammas, color="tab:orange")
axes[0, 1].axvline(K, color="gray", linestyle="--", linewidth=1)
axes[0, 1].set_title("Gamma")
axes[0, 1].set_xlabel("Spot")

axes[1, 0].plot(spots, vegas, color="tab:green")
axes[1, 0].axvline(K, color="gray", linestyle="--", linewidth=1)
axes[1, 0].set_title("Vega")
axes[1, 0].set_xlabel("Spot")

axes[1, 1].plot(spots, thetas, color="tab:red")
axes[1, 1].axvline(K, color="gray", linestyle="--", linewidth=1)
axes[1, 1].set_title("Theta")
axes[1, 1].set_xlabel("Spot")

plt.tight_layout()
plt.savefig("greeks_vs_spot.png", dpi=150)
print("\nSaved plot to greeks_vs_spot.png")

# ---- 4. Implied vol round-trip check ----
market_price = price(S, K, T, r, sigma, option_type, q)
recovered_sigma = implied_vol(market_price, S, K, T, r, option_type, q)
print("\n=== Implied vol round-trip ===")
print(f"True sigma: {sigma:.4f}  Recovered sigma: {recovered_sigma:.4f}")
