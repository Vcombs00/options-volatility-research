"""
Real-data demo. RUN THIS ON YOUR OWN MACHINE, NOT IN A SANDBOX -- it needs
outbound internet access to Yahoo Finance, which most restricted/CI
environments block.

What this does:
  1. Pulls a live options chain for a ticker.
  2. Prices a real contract with your Black-Scholes, binomial, and Monte
     Carlo engines, and compares all three to the actual market price.
  3. Builds and plots an implied volatility smile for one expiry.
  4. Builds and plots a full IV surface across multiple expiries.

Before running: pip install yfinance (already in requirements.txt).
"""

import numpy as np

from options_pricer import price, greeks, binomial_price, mc_price
from options_pricer.data import get_all_chains, get_chain, list_expirations, mid_price
from options_pricer.iv_surface import build_surface, plot_smile, plot_surface_3d

TICKER = "SPY"       # liquid enough to have clean quotes across many strikes
RISK_FREE_RATE = 0.05  # approximate short-term T-bill rate; refine if you want precision
DIV_YIELD = 0.013      # SPY's approximate trailing dividend yield; 0 is fine for a first pass

# ---- 1. Pull one expiry and price a specific contract against the market ----
expirations = list_expirations(TICKER)
print(f"Available expirations for {TICKER}: {expirations[:8]} ...")

near_expiry = expirations[2]  # skip the very nearest -- often thin/expiring-soon noise
snap = get_chain(TICKER, near_expiry)
print(f"\nSpot: {snap.spot:.2f}   Expiry: {snap.expiry}   T: {snap.T:.4f} years")

# pick the closest-to-ATM strike so the comparison is on a liquid contract
calls = snap.calls.copy()
calls["dist_from_spot"] = (calls["strike"] - snap.spot).abs()
atm_row = calls.sort_values("dist_from_spot").iloc[0]
K = atm_row["strike"]
market_px = mid_price(atm_row)

# To compare pricers meaningfully, first back out the market's own implied
# vol from the quoted price -- then feed that into each pricer so all three
# are being evaluated at the same vol, and any price differences you see
# reflect differences between the pricing METHODS, not differences in vol input.
from options_pricer.black_scholes import implied_vol
market_iv = implied_vol(market_px, snap.spot, K, snap.T, RISK_FREE_RATE, "call", DIV_YIELD)

bs_px = price(snap.spot, K, snap.T, RISK_FREE_RATE, market_iv, "call", DIV_YIELD)
bin_px = binomial_price(snap.spot, K, snap.T, RISK_FREE_RATE, market_iv, N=300,
                         option_type="call", american=True, q=DIV_YIELD)
mc_result = mc_price(snap.spot, K, snap.T, RISK_FREE_RATE, market_iv, "call",
                      q=DIV_YIELD, n_paths=200_000, seed=1)

print(f"\n=== ATM call, strike {K} ===")
print(f"Market mid price:      {market_px:.4f}")
print(f"Market implied vol:    {market_iv:.4f}")
print(f"Black-Scholes price:   {bs_px:.4f}  (should ~match market by construction)")
print(f"Binomial (American):   {bin_px:.4f}  (early-exercise premium: {bin_px - bs_px:.4f})")
print(f"Monte Carlo price:     {mc_result.price:.4f}  (95% CI: {mc_result.ci_95})")

g = greeks(snap.spot, K, snap.T, RISK_FREE_RATE, market_iv, "call", DIV_YIELD)
print(f"\nGreeks at market-implied vol: delta={g.delta:.4f} gamma={g.gamma:.4f} "
      f"vega={g.vega/100:.4f} (per 1% vol) theta={g.theta/365:.4f} (per day)")

# ---- 2. Vol smile for this expiry ----
single_chain_surface = build_surface([snap], r=RISK_FREE_RATE, q=DIV_YIELD, option_type="call")
if not single_chain_surface.empty:
    fig = plot_smile(single_chain_surface, snap.expiry)
    fig.savefig("vol_smile.png", dpi=150)
    print(f"\nSaved vol_smile.png ({len(single_chain_surface)} strikes)")

# ---- 3. Full surface across multiple expiries ----
print("\nFetching multiple expirations for full surface (this makes several network calls)...")
chains = get_all_chains(TICKER, max_expirations=6)
surface = build_surface(chains, r=RISK_FREE_RATE, q=DIV_YIELD, option_type="call")
print(f"Surface built: {len(surface)} points across {surface['expiry'].nunique()} expiries")

fig3d = plot_surface_3d(surface)
fig3d.savefig("vol_surface_3d.png", dpi=150)
print("Saved vol_surface_3d.png")

surface.to_csv("iv_surface_data.csv", index=False)
print("Saved raw surface data to iv_surface_data.csv")
