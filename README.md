# Equity Statistical Arbitrage Research Framework

A quantitative research platform for developing, testing, and validating equity pairs trading strategies. The framework implements the full lifecycle of a statistical arbitrage strategy: universe screening, dynamic hedge ratio estimation, signal generation, regime conditioning, rigorous out-of-sample validation, and multi-pair portfolio construction.

---

## Results at a Glance

These figures come from the 10-year daily backtest on a 45-ticker universe (16 sector ETFs + 35 individual stocks).

### Screening Funnel

```
990 candidate pairs
 |
 +-- Engle-Granger pre-filter (p < 0.10)       104 pass  ( 10.5%)
      |
      +-- Johansen trace confirmation            44 pass  (  4.4%)
           |
           +-- Rolling stability >= 55%           0 pass  (  0.0%)  <-- critical gate
                |
                +-- Relaxed fallback (no stability gate)   8 evaluated
```

The rolling stability filter is the hardest gate. No large-cap pair maintained statistically significant cointegration across 55% or more of rolling annual windows over a full decade. This means apparent cointegration in shorter windows is regime-specific, not structural.

---

### Best Pair Performance (AAPL / XLP)

| Metric | Value | Benchmark / Context |
|--------|-------|---------------------|
| Sharpe Ratio | **0.22** | Acceptable range: > 1.0 for institutional deployment |
| Sortino Ratio | **0.03** | Acceptable range: > 1.5 |
| CAGR | **1.3%** | S&P 500 10yr avg: ~13% annualised |
| Win Rate | **71.8%** | Strong signal quality, but see Profit Factor |
| Profit Factor | **1.35** | Break-even = 1.0; institutional target >= 1.5 |
| Max Drawdown | **-$16,645** | On $100,000 notional per leg |
| Avg Holding Period | **6 days** | Consistent with short-term mean reversion |
| Stop Rate | **0.0%** | Spread always reverted before stop triggered |
| Monte Carlo p-value | **1.00** | p >= 0.05 = not distinguishable from random timing |

> **Interpretation:** The win rate and profit factor look encouraging, but the Sharpe of 0.22 is well below the institutional threshold of 1.0, and a Monte Carlo p-value of 1.00 means the observed returns cannot be distinguished from random trade ordering. The strategy does not have a statistically verifiable edge in the large-cap universe over this period.

---

### Walk-Forward Summary (AAPL / XLP)

Out-of-sample results across six non-overlapping one-year test windows:

| Method | Windows | Profitable | OOS Net PnL | Avg OOS Sharpe |
|--------|---------|------------|-------------|----------------|
| Rolling (fixed 3yr train) | 6 | **3 / 6** | +$49,957 | 0.47 |
| Anchored (expanding train) | 6 | **4 / 6** | +$26,809 | 0.49 |

Training-window Engle-Granger p-values ranged from 0.13 to 0.96, confirming the pair is not reliably cointegrated within its own fitting periods. Positive aggregate OOS PnL exists, but is not attributable to a stable statistical relationship.

---

### Portfolio Summary (8 pairs, $2,000,000 notional)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **0.24** |
| Sortino Ratio | **0.04** |
| CAGR | **0.02%** |
| Max Drawdown | **-$4,658** |
| Win Rate | **63.7%** |
| Profit Factor | **1.33** |
| Kelly Fraction (avg) | **0.08** |
| Total Friction Paid | **$5,698** |
| Turnover | **30 round-trips/yr** |

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

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.12.4 | Runtime |
| NumPy | 1.26.4 | Kalman filter, array operations, Monte Carlo |
| pandas | 2.2.2 | Time series alignment, resampling, trade log |
| statsmodels | 0.14.6 | Engle-Granger, Johansen, ADF, OLS regression |
| SciPy | 1.14.0 | Statistical utilities |
| yfinance | 1.4.1 | Market data (Yahoo Finance), daily + hourly OHLCV |
| Matplotlib | 3.9.1 | All charts and dashboards |
| seaborn | 0.13.2 | Correlation heatmaps, monthly PnL grid |
| OpenBB | 4.7.2 | Extended data platform (equities, ETFs, macro) |

Install all dependencies:

```bash
pip install numpy==1.26.4 pandas==2.2.2 statsmodels==0.14.6 scipy==1.14.0 \
            yfinance==1.4.1 matplotlib==3.9.1 seaborn==0.13.2 openbb==4.7.2
```

Or install latest versions:

```bash
pip install numpy pandas statsmodels scipy yfinance matplotlib seaborn openbb
```

---

## Installation and Usage

**Requirements:** Python 3.10+

```bash
git clone https://github.com/devadathspillai/equity-stat-arb.git
cd equity-stat-arb
pip install numpy pandas statsmodels scipy yfinance matplotlib seaborn openbb
python stat_arb.py
```

All charts and CSV outputs are written to the directory containing the script. Expected runtime: 8-15 minutes (990 pairs screened, rolling stability tests, Monte Carlo).

---

## Configuration

All parameters are defined as constants at the top of `stat_arb.py`.

### Universe
```python
SECTOR_ETFS   = ["XLK", "XLF", "XLE", ...]   # 16 sector ETFs
SECTOR_STOCKS = {                              # 35 individual stocks, 7 sectors
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
WF_TRAIN = 756    # training window (3 years of daily bars)
WF_TEST  = 252    # out-of-sample window (1 year)
WF_STEP  = 252    # step between windows (1 year)
```

### Portfolio
```python
PORT_CAPITAL       = 2_000_000   # total portfolio notional
MAX_KELLY          = 0.25        # Kelly fraction cap
HALF_KELLY         = True        # use half-Kelly for conservatism
SPREAD_CORR_CUTOFF = 0.70        # correlation threshold for size penalty
```

---

## Methodology

### Cointegration Screening

The framework applies a four-stage cascading filter to the full C(n, 2) pair universe:

**Stage 1 - Engle-Granger pre-filter**
All pairs are tested with the Engle-Granger two-step test. Pairs with p-value above 0.10 are discarded. This fast O(n^2) sweep reduces the candidate set by roughly 90%.

**Stage 2 - Johansen confirmation**
Surviving pairs are tested with the Johansen trace statistic (k_ar_diff=1, det_order=0). Only pairs where the trace statistic exceeds the 95% critical value are retained. Agreement between two independent tests substantially reduces false positives.

**Stage 3 - Rolling stability**
The Engle-Granger test is re-run on rolling 252-bar windows (stepped every 21 bars). A pair qualifies only if at least 55% of windows show significant cointegration. This is the most discriminating filter: it rejects pairs whose relationship is regime-specific rather than structural.

**Stage 4 - Half-life filter**
The OU half-life is estimated by regressing the first difference of the spread on its lag. Pairs outside [5, 100] days are excluded. Below the minimum implies noise; above the maximum implies the spread reverts too slowly for daily-bar trading.

### Kalman Filter Hedge Ratio

The hedge ratio is estimated dynamically rather than using a static OLS fit.

State-space model:
```
y(t)    = beta(t) * x(t) + eps(t)     observation noise ~ N(0, R_t)
beta(t) = beta(t-1) + w(t)            process noise     ~ N(0, delta)
```

The observation noise R_t is updated adaptively via EWMA of squared innovations. The process noise `delta` controls how quickly the hedge ratio is permitted to drift. A dynamic beta captures shifts in the pair relationship that a static model would miss.

### Signal Generation

The z-score window is set adaptively as a function of the OU half-life:

```
window = max(10, round(half_life * 0.75))
```

### Transaction Cost Model

```
cost = COMM_PER_SHARE * (shares_A + shares_B) * 2
     + (SLIP_BPS / 10000) * (notional_A + notional_B) * 2
```

### Walk-Forward Validation

Two protocols are implemented. In both cases, the hedge ratio applied during the test period is the static OLS estimate from the corresponding training window only. The Kalman filter is not run on test data, enforcing a clean information barrier.

**Rolling:** Fixed 756-bar training window steps forward one year at a time.

**Anchored:** Training window is anchored at the dataset start and expands by one year at each step.

### Portfolio Construction

```
f* = (p * b - (1 - p)) / b

where:
  p = empirical win rate
  b = avg_win / avg_loss

weight(pair) = (f* / 2) * correlation_penalty
allocation   = weight * (total_capital / n_pairs)
```

Pairs with spread correlation above 0.70 have their allocation reduced proportionally to prevent concentration in structurally similar trades.

### Performance Metrics

| Metric | Description |
|--------|-------------|
| Sharpe | Annualized: daily PnL mean / std * sqrt(252) |
| Sortino | Annualized using downside deviation only |
| Calmar | CAGR / max drawdown |
| Max Drawdown | Peak-to-trough decline on cumulative PnL |
| Max DD Duration | Longest consecutive drawdown in trading days |
| Win Rate | Fraction of trades closing with positive net PnL |
| Payoff Ratio | avg_win / avg_loss |
| Profit Factor | sum(wins) / sum(losses) |
| Kelly Fraction | Optimal position size per Kelly formula (half-Kelly applied) |
| VaR 95% | Value at Risk at 95% confidence on daily PnL |
| CVaR 95% | Expected shortfall beyond VaR |
| Avg Holding Period | Mean trade duration in calendar days |
| Stop Rate | Fraction of trades exiting via stop-loss |
| Turnover | Round-trips per year |
| MC p-value | Fraction of Monte Carlo shuffles exceeding observed Sharpe |

---

## Output Files

All outputs are written to the same directory as the script.

| File | Description |
|------|-------------|
| `01_screening_landscape.png` | EG p-value distribution, stability vs cointegration scatter, half-life chart, test agreement |
| `02_dashboard_{A}_{B}.png` | 8-panel pair dashboard: prices, Kalman beta, spread, z-score, equity curve, holding periods, MAE/MFE, per-trade PnL |
| `03_wf_{A}_{B}.png` | Rolling vs anchored walk-forward: OOS PnL, Sharpe, and equity curve per window |
| `04_portfolio_dashboard.png` | Portfolio equity, drawdown, rolling Sharpe, pair contributions, Kelly weights, monthly PnL heatmap |
| `all_pairs_screened.csv` | Full results for all EG+Johansen survivors |
| `qualified_pairs.csv` | Pairs passing the full four-stage filter |
| `wf_rolling_{A}_{B}.csv` | Per-window rolling walk-forward metrics |
| `wf_anchored_{A}_{B}.csv` | Per-window anchored walk-forward metrics |
