# Options Pricing, Volatility Research, and Backtesting Engine

**A from-scratch quantitative research project: pricing engines, an event-driven
backtester, and an empirical study of the equity volatility risk premium.**

---

## 1. Motivation

Options pricing and volatility trading sit at the center of a lot of quantitative
finance work, but most student projects in this space stop at "I implemented
Black-Scholes." This project was built to go further: implement multiple
independent pricing methods and prove they agree, build a real backtesting
engine with correct accounting, confront the actual data-availability problem
that any options researcher without a paid vendor runs into, and use a real
(free) data source to test an actual empirical hypothesis about volatility
markets — the volatility risk premium — rather than stopping at infrastructure.

The guiding principle throughout: every claim in this write-up is backed by a
test I actually ran and can reproduce, not by a plausible-sounding argument.
Several of my own hypotheses below turned out to be wrong when tested, and
I've kept those in rather than editing them out, because the process of
proposing an explanation, testing it, and discarding it when the data disagreed
is a more accurate picture of how this kind of research actually works than
a write-up that only shows the ideas that panned out.

## 2. System Overview

The project has three layers, each building on the last:

| Layer | Contents |
|---|---|
| **Pricing engine** | Black-Scholes (closed-form, with full Greeks), a Cox-Ross-Rubinstein binomial tree (American exercise), and a Monte Carlo pricer (GBM, antithetic variates) |
| **Data access** | Live options chains and spot prices via yfinance; historical VIX (CBOE's implied vol index) as a free proxy for real market implied vol |
| **Backtesting engine** | Event-driven day-by-day simulation, portfolio accounting (cash/stock/options, mark-to-market), and four strategies of increasing complexity |

Full source is organized as:

```
options_pricer/   black_scholes.py, binomial.py, monte_carlo.py, data.py, iv_surface.py
backtest/         portfolio.py, engine.py, strategies.py, metrics.py
```

## 3. Pricing Engine

### 3.1 Methods implemented

- **Black-Scholes-Merton**, closed-form, with a continuous dividend yield term
  (so the same formula covers dividend-paying equities, FX, and futures
  options as special cases). Full analytic Greeks (delta, gamma, vega, theta,
  rho) are implemented from their own closed-form derivatives rather than via
  numerical differentiation, since exact analytic Greeks are both faster and
  more precise than bump-and-reprice when a closed form exists.
- **Binomial tree (CRR)**, supporting both European and American exercise.
  This exists specifically because Black-Scholes cannot price American
  options, and most listed US equity options are American-style.
- **Monte Carlo**, simulating terminal GBM prices directly (exact, since
  European payoffs depend only on the terminal price — no time-discretization
  error), with antithetic variates for variance reduction.
- **Implied volatility solver**: Newton-Raphson using vega as the derivative,
  falling back to bisection when vega is too small for Newton to be stable
  (deep ITM/OTM or very short-dated contracts).

### 3.2 Validation

Rather than trust any one method, each was cross-checked against the others:

- The binomial tree (N=500 steps, European mode) converges to the
  Black-Scholes price to within ~0.3 cents on a representative contract
  ($6.8887 vs. $6.8859).
- The American call price from the binomial tree exactly equals the European
  price for a non-dividend underlier — the correct theoretical result, since
  early exercise of a call is never optimal without dividends.
- Monte Carlo (500,000 paths) lands within its own 95% confidence interval
  of the Black-Scholes price, and put-call parity (`C - P = S·e^(-qT) - K·e^(-rT)`)
  holds to 6 decimal places.
- The implied vol solver exactly recovers a known input volatility from a
  price generated at that volatility (round-trip test).

## 4. The Data Problem, and How It Was Solved

Free historical *options chains* — the raw material needed to backtest an
options strategy against real market prices — do not exist. Paid vendors
(ORATS, ivolatility, Cboe DataShop, Databento) fill this gap commercially,
but that wasn't a viable option here.

**First approach (and its known limitation):** price options in the backtest
using a rolling realized volatility of the underlying, computed from free
historical stock prices, instead of real implied vol. This is a standard,
explicit simplification — but it has a specific consequence worth stating
plainly: it means the backtest has **no built-in volatility risk premium**,
since the same number is used both to price options and to simulate how the
world evolves. This was validated directly: with zero transaction costs, an
unconditional short-straddle strategy priced this way converges to breakeven
(+0.39% over a ~2-year synthetic test), exactly as theory predicts when there
is no edge to extract.

**Second approach (the actual fix):** CBOE's VIX index is a free, decades-deep
history of the market's own 30-day implied volatility on the S&P 500. Feeding
real historical VIX into the pricing engine as the "pricing vol," while
letting the underlying (SPY) evolve at its own true realized vol, reproduces
the situation a real desk actually faces — and is the mechanism that lets a
genuine volatility risk premium show up in a backtest, without needing paid
options data at all. This only works for index/SPY-level strategies (VIX
doesn't cover individual names), which is an explicit scope limitation, not
an oversight.

This was also validated before trusting it on real data: feeding in a
synthetic vol series set to 1.15x realized vol (a controlled, artificially
"rich" IV) produced a clearly positive result (+8.9% over the same period,
vs. the +0.39% breakeven case with no premium) — confirming the mechanism
actually transmits an edge into the P&L rather than being silently broken.

## 5. Backtesting Engine

Event-driven: the simulation processes one trading day at a time, and a
strategy can only act on information known as of that day — eliminating
lookahead bias by construction, which is the most common way retail
backtests silently generate fake returns.

Four strategies were implemented:

1. **Buy & Hold** — benchmark, no options.
2. **Covered Call** — sell one ~5%-OTM call per month against a held stock
   position; static.
3. **Delta-Hedged Short Straddle** — sell an ATM call + put each cycle, then
   trade the underlying daily to keep net option delta near zero, isolating
   a bet on volatility from a bet on direction. Sells unconditionally, every
   cycle.
4. **VRP-Conditional Short Straddle** — identical, but only opens a new
   position when the market's implied vol (VIX) exceeds trailing realized
   vol by a chosen threshold — testing whether a real, literature-standard
   signal for timing vol sales actually helps.

## 6. Results on Real Data

All numbers below are from actual SPY and VIX price history, not synthetic
data. All four backtests used a ~2-to-5-year historical window ending
August 2026.

### 6.1 Strategy comparison, ~2-year window (2024-08-07 to 2026-08-06)

Options priced using realized vol (no VRP edge — a directional/static-strategy
comparison):

| Strategy | Total Return | CAGR | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|---|
| Buy & Hold | 30.45% | 14.30% | 11.91% | 0.76 | -13.69% |
| Covered Call | 27.21% | 12.87% | 9.80% | 0.78 | -12.96% |
| Delta-Hedged Short Straddle | -2.50% | -1.26% | 5.37% | -1.14 | -8.83% |

SPY rose from $506.67 to $768.56 over this window — a strong bull run. The
covered call gave up some upside for meaningfully lower volatility and
drawdown, as expected. The straddle, priced without any implied-vol edge,
lost money — consistent with the zero-edge design (see Section 4).

### 6.2 The volatility risk premium, measured directly (5-year window)

Using real VIX and SPY history (2021-08-09 to 2026-08-06, 1,254 trading days):

**VIX exceeded trailing 20-day realized SPY volatility on 85.1% of trading
days**, by an average of 3.62 vol points (median 3.80). This is a direct,
real-data confirmation of the volatility risk premium — implied vol
genuinely runs richer than what subsequently realizes, most of the time,
over this window.

Pricing the straddle strategies at real VIX levels:

| Strategy | Total Return | CAGR | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|---|
| Unconditional Short Straddle | 39.00% | 6.84% | 3.38% | 0.50 | -3.32% |
| VRP-Conditional (3-pt threshold) | 33.38% | 5.96% | 3.82% | 0.23 | -5.18% |

Both strategies were profitable with low volatility relative to buy-and-hold
— consistent with harvesting a real, persistent premium. But the conditional
filter, despite selecting for a stronger signal, **underperformed the
unconditional version on a risk-adjusted basis.** This surprising result is
the central finding investigated in the rest of this section.

### 6.3 Investigating why the conditional filter underperformed

**Threshold sweep.** Running the VRP-conditional strategy across a range of
thresholds (0 to 8 vol points) showed performance degrading roughly
monotonically as the threshold rose:

| Threshold (vol pts) | Trades | Total Return | Sharpe | Max DD |
|---|---|---|---|---|
| Unconditional | 59 | 39.00% | 0.50 | -3.32% |
| 0 | 56 | 39.18% | 0.49 | -4.34% |
| 1 | 55 | 37.14% | 0.40 | -4.36% |
| 2 | 52 | 31.19% | 0.14 | -6.36% |
| 3 | 51 | 33.38% | 0.23 | -5.18% |
| 5 | 40 | 25.36% | -0.11 | -3.90% |
| 8 | 24 | 21.31% | -0.32 | -3.25% |

More selectivity made things worse, not better. Two hypotheses were formed
and tested in turn.

**Hypothesis 1 (rejected): the filter concentrates trades into crisis
periods.** Since VRP tends to spike during market stress, maybe a higher
threshold disproportionately selects entries clustered around a small number
of volatile, high-tail-risk windows. Tested directly by comparing the spread
(standard deviation) of trade entry dates between the top and bottom half of
trades by VRP-at-entry, on the 51 trades at the 3-point threshold:

| Bucket | Trades | Avg P&L | Win Rate | Entry-Date Spread (days) |
|---|---|---|---|---|
| High VRP (above median) | 26 | $447.78 | 80.8% | 547.9 |
| Low VRP (below median) | 25 | $347.09 | 80.0% | 476.1 |

High-VRP trades were spread out *more*, not less, than low-VRP trades. This
hypothesis is directly contradicted by the data and was discarded.

**Hypothesis 2 (rejected): high-VRP trades are individually riskier, offsetting
their higher average P&L.** Tested by comparing per-trade P&L variance
between the same two buckets:

| Bucket | Mean | Std Dev | Sum | Count |
|---|---|---|---|---|
| High VRP | $447.78 | $608.48 | $11,642.20 | 26 |
| Low VRP | $347.09 | $562.62 | $8,677.15 | 25 |

Per-trade risk is only modestly higher for High VRP (~8%), and the
mean-to-std ratio is actually *better* for High VRP (0.74 vs. 0.62) — so
these trades are not lower quality on a risk-adjusted basis either. This
hypothesis was also discarded.

**Actual explanation: opportunity cost of a binary filter.** The correlation
between VRP-at-entry and trade P&L across the full 51-trade sample is a real
but modest +0.338 — richer entries genuinely do earn more, on average. But
the strategy's entry rule is binary: sell one full-size straddle, or hold
nothing at all. Since VIX exceeded realized vol on 85% of days in this
sample, "elevated VRP" was closer to the norm than a rare, special
condition. Raising the threshold doesn't isolate a small number of
exceptional opportunities — it excludes trades that were still solidly
profitable (an 80.0% win rate in the *excluded* low-VRP bucket, nearly
identical to the 80.8% win rate of the included trades) in favor of sitting
in cash. The modest per-trade quality gain from filtering is outweighed by
the cost of the good trades given up. This is a specific, testable, and
here-confirmed instance of a general trap in naive signal-filtering: a
signal can be real and directionally correct while a binary gate built on it
still hurts aggregate performance, because the gate ignores the expected
value of what it excludes.

**Implication for a redesign.** A binary gate is the wrong shape for a
weak-but-real, mostly-present signal like this one. Sizing the position
continuously with the magnitude of the VRP — smaller size in weak-signal
periods rather than zero — would keep the strategy exposed to the reliably
positive baseline case while still tilting risk toward higher-conviction
entries. This is a concrete next step, not yet implemented, and is the
correct fix implied by the diagnosis above.

## 7. Limitations

Stated explicitly rather than glossed over:

- **Single historical window.** All real-data results above come from one
  ~2-to-5-year historical period ending August 2026. None of this has been
  validated out-of-sample or across multiple independent windows — the
  threshold sweep, in particular, was run and interpreted on the same data
  it was evaluated on, which is a form of look-ahead in the broad sense (not
  a lookahead bug in the engine, but a methodological one in how the
  research question was posed). A walk-forward split is the natural fix.
- **European pricing for American options.** All options are marked to
  market using Black-Scholes rather than the binomial tree in the backtest,
  for computational speed. Justified for calls without dividends (proven in
  Section 3.2); weaker for puts, where early exercise can occasionally be
  optimal.
- **VIX ≠ SPY options specifically.** VIX reflects SPX (index) option
  pricing; close to but not identical to SPY's own options market.
- **No margin modeling.** Short option positions can push simulated cash
  negative, treated as implicit zero-cost borrowing. A real short-vol book
  has margin requirements that constrain position sizing and would change
  the realistic capital efficiency of these results.
- **Flat transaction cost model.** Costs are a simple bps-of-notional charge,
  not a realistic bid/ask spread plus per-contract commission structure.
- **Individual-name strategies are still untested**, since VIX only provides
  an index-level proxy; single-stock strategies would need either paid data
  or a self-collected implied-vol history built up over time.

## 8. What This Project Demonstrates

- Independent implementation and cross-validation of three separate options
  pricing methods, with every claim of correctness backed by a specific
  numerical check rather than trust in the formula being "well known."
- Recognition of a real data-availability constraint (no free historical
  options chains) and a specific, defensible, tested solution (VIX as an
  implied-vol proxy) rather than either ignoring the problem or faking
  results.
- A correctly-built event-driven backtest with verified absence of
  lookahead bias and a portfolio accounting layer validated against
  known-zero and known-positive edge cases before being trusted on real data.
- An actual empirical finding — VIX exceeded realized SPY vol on 85% of days
  over five years, and naive threshold-filtering on that premium hurts
  risk-adjusted returns via an opportunity-cost mechanism, not a signal-quality
  one — arrived at through a hypothesize-test-discard cycle where two
  plausible initial explanations were tested and rejected before finding
  the one the data actually supported.

## 9. Reproducing These Results

```bash
pip install -r requirements.txt

python demo.py                    # pricing engine validation, no internet needed
python run_backtest.py            # strategy comparison, real data if available
python vrp_backtest.py            # VIX-based VRP measurement and straddle test
python vrp_threshold_sweep.py     # threshold sweep + trade-level breakdown
python trade_timeline_diagnostic.py  # clustering/quality hypothesis tests
```

Full source, including the pricing engine, backtest infrastructure, and all
validation checks referenced above, is organized under `options_pricer/` and
`backtest/`.
