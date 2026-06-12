# Equity Statistical Arbitrage Research Framework

A quantitative research platform for developing, testing, and validating equity pairs trading strategies. The framework implements the full lifecycle of a statistical arbitrage strategy: universe screening, dynamic hedge ratio estimation, signal generation, regime conditioning, rigorous out-of-sample validation, and multi-pair portfolio construction.

---

## Table of Contents

- [Overview](#overview)
- [Strategy Logic](#strategy-logic)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Methodology](#methodology)
  - [Cointegration Screening](#cointegration-screening)
  - [Kalman Filter Hedge Ratio](#kalman-filter-hedge-ratio)
  - [Signal Generation](#signal-generation)
  - [Transaction Cost Model](#transaction-cost-model)
  - [Walk-Forward Validation](#walk-forward-validation)
  - [Portfolio Construction](#portfolio-construction)
  - [Analytics](#analytics)
- [Key Findings](#key-findings)
- [Output Files](#output-files)
- [Dependencies](#dependencies)

---

## Overview

Statistical arbitrage exploits temporary price divergences between cointegrated equity pairs. When two securities share a long-run equilibrium relationship, short-term deviations from that relationship are expected to revert, creating a tradeable spread.

This framework implements:

1. Universe screening via a cascading four-stage cointegration filter
2. Dynamic hedge ratio estimation using a Kalman filter
3. Signal generation based on the Ornstein-Uhlenbeck half-life of the spread
4. Regime-conditional trade execution (suppressed in high-volatility environments)
5. Rigorous out-of-sample validation via rolling and anchored walk-forward
6. Kelly-weighted multi-pair portfolio construction with spread correlation management
7. 20+ performance metrics including VaR, CVaR, MAE/MFE, and Monte Carlo significance

---

## Strategy Logic

```
For each pair (A, B):
  1. Estimate dynamic hedge ratio beta(t) via Kalman filter
  2. Compute spread:  s(t) = P_A(t) - beta(t) * P_B(t)
  3. Standardize:     z(t) = (s(t) - mu) / sigma   (window = f(OU half-life))
  4. Enter long A / short B  when  z(t) < -ENTRY_Z  (spread below mean)
     Enter short A / long B  when  z(t) > +ENTRY_Z  (spread above mean)
  5. Exit when z(t) reverts to +/-EXIT_Z
  6. Stop-loss when |z(t)| >= STOP_Z
  7. Skip entries when the spread is in a high-volatility regime
```

---

## Installation

**Requirements:** Python 3.10+

```bash
pip install numpy pandas statsmodels scipy yfinance matplotlib seaborn openbb
```

No API keys are required. Market data is sourced from Yahoo Finance via yfinance.

---

## Usage

```bash
python stat_arb.py
```

All charts and CSV outputs are written to the same directory as the script. Expected runtime: 8-15 minutes (990 pairs screened across a 45-ticker universe, rolling stability tests, Monte Carlo iterations).

---

## Configuration

All parameters are defined as constants near the top of `stat_arb.py`.

### Universe
```python
SECTOR_ETFS   = ["XLK", "XLF", "XLE", ...]   # 16 sector ETFs
SECTOR_STOCKS = {                              # 35 individual stocks across 7 sectors
    "Financials":  ["JPM", "BAC", "WFC", ...],
    "Tech":        ["AAPL", "MSFT", "GOOGL", ...],
    ...
}
DATA_PERIOD   = "10y"
```

### Cointegration Screening
```python
EG_PREFILTER_PVAL  = 0.10    # generous first-pass threshold
EG_PVAL_THRESHOLD  = 0.05    # strict qualification threshold
STABILITY_WINDOW   = 252     # rolling window for stability test (trading days)
STABILITY_MIN_PCT  = 0.55    # minimum fraction of windows showing cointegration
HALF_LIFE_MIN      = 5       # minimum OU half-life (days)
HALF_LIFE_MAX      = 100     # maximum OU half-life (days)
```

### Strategy
```python
ENTRY_Z  = 2.0      # z-score level to open a position
EXIT_Z   = 0.5      # z-score level to close
STOP_Z   = 3.5      # z-score stop-loss
CAPITAL  = 100_000  # notional per leg (USD)
```

### Kalman Filter
```python
KALMAN_DELTA = 1e-4    # process noise (controls how quickly beta can drift)
KALMAN_VE    = 0.001   # initial observation noise variance
```

### Transaction Costs
```python
COMM_PER_SHARE = 0.005   # $0.005 per share (Interactive Brokers retail)
SLIP_BPS       = 5       # 5 basis points per side
```

### Walk-Forward
```python
WF_TRAIN = 756    # training window (3 years)
WF_TEST  = 252    # out-of-sample test window (1 year)
WF_STEP  = 252    # step size between windows (1 year)
```

### Portfolio
```python
PORT_CAPITAL       = 2_000_000   # total portfolio notional
MAX_KELLY          = 0.25        # Kelly fraction cap
HALF_KELLY         = True        # use half-Kelly for conservatism
SPREAD_CORR_CUTOFF = 0.70        # correlation above which to penalize sizing
```

---

## Methodology

### Cointegration Screening

The framework applies a four-stage cascading filter to identify tradeable pairs from the full C(n, 2) universe:

**Stage 1 - Engle-Granger pre-filter**
All pairs are tested with the Engle-Granger two-step test. Pairs with p-value above 0.10 are discarded. This fast O(n^2) sweep reduces the candidate set by roughly 90%.

**Stage 2 - Johansen confirmation**
Surviving pairs are tested with the Johansen trace statistic. Only pairs where the trace statistic exceeds the 95% critical value are retained. Agreement between two independent tests substantially reduces false positives.

**Stage 3 - Rolling stability**
The Engle-Granger test is re-run on rolling 252-bar windows (stepped every 21 bars). A pair qualifies only if at least 55% of windows show significant cointegration. This is the most discriminating filter: it rejects pairs whose relationship is regime-specific rather than structural.

**Stage 4 - Half-life filter**
The OU half-life is estimated by regressing the first difference of the spread on its lag. Pairs with half-life below 5 or above 100 days are excluded. Below the minimum implies noise; above the maximum implies the spread reverts too slowly to be actionable on daily bars.

### Kalman Filter Hedge Ratio

The hedge ratio is estimated dynamically using a scalar Kalman filter.

State-space model:
```
y(t)    = beta(t) * x(t) + eps(t)     observation noise ~ N(0, R_t)
beta(t) = beta(t-1) + w(t)            process noise     ~ N(0, delta)
```

The observation noise variance R_t is updated adaptively at each bar via EWMA of squared innovations. This allows the filter to self-tune to changing spread volatility without manual recalibration. The process noise `delta` controls how quickly the hedge ratio is permitted to drift between observations.

Using a dynamic hedge ratio rather than static OLS captures changes in the pair relationship over time - shifts in relative market cap, changes in index composition, or post-event repricing.

### Signal Generation

The z-score normalization window is set adaptively as a function of the estimated OU half-life:

```
window = max(10, round(half_life * 0.75))
```

This keeps the normalization window proportional to the mean-reversion speed, avoiding over-smoothing for slow-reverting pairs and under-smoothing for fast-reverting ones.

### Transaction Cost Model

Each round-trip trade incurs costs on both legs, on both entry and exit:

```
cost = COMM_PER_SHARE * (shares_A + shares_B) * 2
     + (SLIP_BPS / 10000) * (notional_A + notional_B) * 2
```

A sensitivity sweep across commission levels ($0 to $0.02/share) and slippage (0 to 20 bps) is available to quantify how robust a given pair's edge is to execution friction.

### Walk-Forward Validation

Two walk-forward protocols are implemented:

**Rolling:** Fixed-size training window (756 bars) steps forward one year at a time. Parameters are re-estimated from scratch on each slice. The strategy is evaluated on the following year using those frozen parameters.

**Anchored:** Training window is anchored at the start of history and expands by one year at each step. Tests whether performance improves as more data accumulates.

In both protocols, the hedge ratio applied during the test period is the static OLS estimate from the corresponding training period. The Kalman filter is not run on test data. This enforces a clean information barrier between fitting and evaluation.

### Portfolio Construction

Kelly criterion is applied to size positions across all qualified pairs:

```
f* = (p * b - (1 - p)) / b

where:
  p = empirical win rate
  b = avg_win / avg_loss (payoff ratio)
```

Half-Kelly is used by default to account for estimation error. If two pair spreads have correlation above 0.70, the lower-Kelly pair's allocation is reduced by a factor proportional to the correlation, preventing concentration in structurally similar trades.

### Analytics

The following metrics are computed for each pair and for the aggregate portfolio:

| Metric | Description |
|--------|-------------|
| Sharpe | Annualized: daily PnL mean / std * sqrt(252) |
| Sortino | Annualized using downside deviation only |
| Calmar | CAGR / max drawdown |
| Max Drawdown | Peak-to-trough decline on cumulative PnL |
| Max DD Duration | Longest consecutive drawdown in trading days |
| Win Rate | Fraction of trades closing with positive net PnL |
| Payoff Ratio | avg_win / avg_loss |
| Profit Factor | sum of wins / sum of losses |
| Kelly Fraction | Optimal position size fraction (half-Kelly applied) |
| VaR 95% | Value at Risk at 95% confidence on daily PnL distribution |
| CVaR 95% | Expected shortfall beyond the VaR threshold |
| Avg Holding Period | Mean trade duration in calendar days |
| Stop Rate | Fraction of trades exiting via stop-loss rather than mean reversion |
| Turnover | Round-trips per year |
| MC p-value | Fraction of Monte Carlo shuffles that exceed the observed Sharpe |

---

## Key Findings

Results from the 10-year daily backtest on a 45-ticker universe (16 ETFs + 35 stocks).

### Cointegration Screening

| Stage | Pairs Remaining |
|-------|----------------|
| Engle-Granger pre-filter (p < 0.10) | 104 / 990 |
| + Johansen confirmation | 44 / 990 |
| + Rolling stability (>=55% of windows) | 0 / 990 |
| Relaxed fallback (EG + Johansen, no stability gate) | 8 evaluated |

No pair in the large-cap universe maintained significant cointegration across 55% or more of rolling annual windows over the full 10-year period. This indicates that cointegration observed in shorter windows is regime-specific rather than structural.

### Backtest Results (relaxed filter, 8 pairs)

| Pair | Sharpe | Sortino | CAGR | Win Rate | Profit Factor | MC p-value |
|------|--------|---------|------|----------|---------------|-----------|
| AAPL/XLP | 0.22 | 0.03 | 1.3% | 71.8% | 1.35 | 1.00 |
| AAPL/KO | 0.15 | 0.03 | 1.7% | 54.7% | 1.23 | 1.00 |

MC p-values of 1.00 indicate the observed Sharpe ratios are statistically indistinguishable from random trade timing. No pair achieved a Sharpe above 0.25 after applying regime conditioning and transaction costs.

### Walk-Forward (AAPL/XLP, best Sharpe pair)

| Method | Windows | Profitable Windows | Aggregate OOS PnL |
|--------|---------|-------------------|------------------|
| Rolling | 6 | 3 / 6 | +$49,957 |
| Anchored | 6 | 4 / 6 | +$26,809 |

Training-window EG p-values ranged from 0.13 to 0.96 across all windows, confirming the pair does not exhibit stable cointegration in its own fitting periods. Any positive OOS PnL is not attributable to a reliable statistical relationship.

### Research Conclusion

Large-cap US equity cointegration is not a durable structural phenomenon over 10-year horizons. The rolling stability filter is the critical discriminating test: pairs that appear cointegrated in a given 2-year window frequently fail the relationship in adjacent windows across different market regimes.

The framework correctly identifies that in-sample results from short lookbacks overstate the robustness of the strategy, and provides the tooling to quantify how and when relationships break down.

---

## Output Files

All outputs are written to the same directory as the script.

| File | Description |
|------|-------------|
| `01_screening_landscape.png` | EG p-value distribution, stability vs cointegration scatter, half-life chart, test agreement |
| `02_dashboard_{A}_{B}.png` | 8-panel pair dashboard: prices, Kalman beta, spread, z-score, equity curve, holding periods, MAE/MFE, per-trade PnL |
| `03_wf_{A}_{B}.png` | Rolling vs anchored walk-forward: OOS PnL, Sharpe, and equity curve per window |
| `04_portfolio_dashboard.png` | Portfolio equity, drawdown, rolling Sharpe, pair contributions, Kelly weights, monthly PnL heatmap |
| `all_pairs_screened.csv` | Full screening results for all EG+Johansen survivors |
| `qualified_pairs.csv` | Pairs passing the full four-stage filter |
| `wf_rolling_{A}_{B}.csv` | Per-window rolling walk-forward metrics |
| `wf_anchored_{A}_{B}.csv` | Per-window anchored walk-forward metrics |

---

## Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| numpy | 1.26 | Numerical computation, Kalman filter |
| pandas | 2.0 | Time series manipulation |
| statsmodels | 0.14 | Engle-Granger, Johansen, ADF, OLS |
| scipy | 1.12 | Statistical utilities |
| yfinance | 0.2 | Market data via Yahoo Finance |
| matplotlib | 3.8 | Charts and dashboards |
| seaborn | 0.13 | Correlation heatmaps |
| openbb | 4.0 | Extended data platform (optional) |
