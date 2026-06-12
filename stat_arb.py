"""
Equity Statistical Arbitrage Research Framework
================================================
Quantitative pairs trading strategy for US equities.

Pipeline:
  1. Universe screening  - Engle-Granger + Johansen cascade, rolling stability,
                           Ornstein-Uhlenbeck half-life filter
  2. Signal generation   - Kalman filter dynamic hedge ratio, adaptive z-score window
  3. Regime conditioning - volatility regime filter (skip high-RV environments)
  4. Backtesting         - full transaction cost model (commission + slippage)
  5. Validation          - rolling and anchored walk-forward, Monte Carlo significance
  6. Portfolio           - Kelly-weighted multi-pair with spread correlation penalty
  7. Risk               - Sharpe, Sortino, Calmar, VaR, CVaR, MAE/MFE, market beta

Usage:
  python stat_arb.py

All outputs (charts, CSVs) are written to the directory containing this script.
Configuration constants are at the top of the file.
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import yfinance as yf
from datetime import datetime

np.random.seed(42)

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------

OUT = os.path.dirname(os.path.abspath(__file__))

SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLB", "XLU", "XLP", "XLY",
    "XLRE", "QQQ", "SPY", "IWM", "GLD", "TLT", "HYG",
]

SECTOR_STOCKS = {
    "Financials":  ["JPM", "BAC", "WFC", "GS", "MS"],
    "Energy":      ["XOM", "CVX", "COP", "SLB"],
    "Tech":        ["AAPL", "MSFT", "GOOGL", "META"],
    "Retail":      ["HD", "LOW", "WMT", "TGT"],
    "Consumer":    ["KO", "PEP", "MCD", "SBUX"],
    "Healthcare":  ["JNJ", "UNH", "PFE", "ABBV"],
    "Industrials": ["CAT", "HON", "GE", "MMM"],
}

DATA_PERIOD   = "10y"
DATA_INTERVAL = "1d"

# Screening
EG_PREFILTER_PVAL   = 0.10   # generous first pass
EG_PVAL_THRESHOLD   = 0.05   # strict qualification
STABILITY_WINDOW    = 252
STABILITY_STEP      = 21
STABILITY_MIN_PCT   = 0.55   # 55% of rolling windows cointegrated
HALF_LIFE_MIN       = 5
HALF_LIFE_MAX       = 100

# Strategy
ENTRY_Z = 2.0
EXIT_Z  = 0.5
STOP_Z  = 3.5
CAPITAL = 100_000

# Kalman
KALMAN_DELTA = 1e-4
KALMAN_VE    = 0.001

# Walk-forward (daily bars)
WF_TRAIN = 756   # 3 years
WF_TEST  = 252   # 1 year
WF_STEP  = 252

# Transaction costs
COMM_PER_SHARE = 0.005
SLIP_BPS       = 5

# Portfolio
PORT_CAPITAL       = 2_000_000
MAX_KELLY          = 0.25
HALF_KELLY         = True
SPREAD_CORR_CUTOFF = 0.70

# Regime
REGIME_WINDOW   = 63
REGIME_PCT_HIGH = 0.70    # top 30% RV = high-vol regime
SKIP_HIGH_REGIME = True

MC_ITERS = 500

# -----------------------------------------------------------------
# SECTION 1 - DATA
# -----------------------------------------------------------------

def build_universe():
    tickers = set(SECTOR_ETFS)
    for stocks in SECTOR_STOCKS.values():
        tickers.update(stocks)
    return sorted(tickers)

def fetch_prices(tickers, period=DATA_PERIOD, interval=DATA_INTERVAL):
    raw = yf.download(tickers, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    prices = (raw["Close"] if len(tickers) > 1 else raw.rename(columns={raw.columns[0]: tickers[0]}))
    prices = prices.dropna(how="all", axis=1)
    min_bars = 252 * 4
    prices = prices.loc[:, prices.notna().sum() >= min_bars]
    prices = prices.ffill().dropna()
    return prices

# -----------------------------------------------------------------
# SECTION 2 - COINTEGRATION BATTERY
# -----------------------------------------------------------------

def eg_test(s1, s2):
    try:
        _, p, _ = coint(s1, s2)
        return float(p)
    except Exception:
        return 1.0

def johansen_test(df, ta, tb):
    try:
        data = df[[ta, tb]].values
        res  = coint_johansen(data, det_order=0, k_ar_diff=1)
        trace = res.lr1[0]
        cv95  = res.cvt[0, 1]
        evec  = res.evec[:, 0]
        jo_beta = -evec[0] / evec[1] if abs(evec[1]) > 1e-9 else None
        return {"jo_trace": round(trace, 3), "jo_cv95": round(cv95, 3),
                "jo_pass": bool(trace > cv95), "jo_beta": jo_beta}
    except Exception:
        return {"jo_trace": 0, "jo_cv95": 999, "jo_pass": False, "jo_beta": None}

def ols_beta(df, ta, tb):
    y = df[ta].values
    x = add_constant(df[tb].values)
    return float(OLS(y, x).fit().params[1])

def half_life(spread: pd.Series) -> float:
    s = spread.values
    ds = np.diff(s)
    sl = s[:-1]
    X  = np.column_stack([np.ones(len(sl)), sl])
    try:
        b = np.linalg.lstsq(X, ds, rcond=None)[0][1]
        if b >= 0:
            return float("inf")
        hl = float(-np.log(2) / np.log(1 + b))
        return float(np.clip(hl, HALF_LIFE_MIN, HALF_LIFE_MAX))
    except Exception:
        return float("inf")

def rolling_stability(df, ta, tb):
    n, sig, total = len(df), 0, 0
    i = 0
    while i + STABILITY_WINDOW <= n:
        sl = df.iloc[i : i + STABILITY_WINDOW]
        p  = eg_test(sl[ta].values, sl[tb].values)
        sig   += int(p < EG_PVAL_THRESHOLD)
        total += 1
        i     += STABILITY_STEP
    return sig / total if total > 0 else 0.0

def screen_pairs(prices):
    tickers = list(prices.columns)
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    print(f"  Screening {n_pairs} pairs (cascade: EG -> Johansen -> Stability -> HL)")

    stage1, stage2, qualified = [], [], []

    # Stage 1: fast EG pre-filter
    for ta, tb in combinations(tickers, 2):
        p = eg_test(prices[ta].values, prices[tb].values)
        if p < EG_PREFILTER_PVAL:
            stage1.append((ta, tb, p))

    print(f"  Stage 1 (EG p<{EG_PREFILTER_PVAL}): {len(stage1)} pairs")

    # Stage 2: Johansen on survivors
    for ta, tb, eg_p in stage1:
        jo = johansen_test(prices, ta, tb)
        stage2.append((ta, tb, eg_p, jo))

    jo_pass = [(ta, tb, eg_p, jo) for ta, tb, eg_p, jo in stage2 if jo["jo_pass"]]
    print(f"  Stage 2 (+ Johansen):  {len(jo_pass)} pairs")

    # Stage 3 + 4: stability + half-life (only on EG+Johansen survivors)
    records = []
    for ta, tb, eg_p, jo in jo_pass:
        beta = ols_beta(prices, ta, tb)
        spr  = prices[ta] - beta * prices[tb]
        hl   = half_life(spr)
        if hl < HALF_LIFE_MIN or hl > HALF_LIFE_MAX:
            continue
        adf_p = float(adfuller(spr.dropna())[1])
        stab  = rolling_stability(prices, ta, tb) if eg_p < EG_PVAL_THRESHOLD else 0.0

        records.append({
            "ta": ta, "tb": tb,
            "eg_p":    round(eg_p, 4),
            "jo_pass": jo["jo_pass"],
            "jo_trace":jo["jo_trace"],
            "jo_cv95": jo["jo_cv95"],
            "adf_p":   round(adf_p, 4),
            "hl":      round(hl, 1),
            "stab":    round(stab, 3),
            "beta":    round(beta, 4),
        })

    all_df = pd.DataFrame(records)

    if all_df.empty:
        return pd.DataFrame(), all_df

    mask = (
        (all_df["eg_p"]   <  EG_PVAL_THRESHOLD) &
        (all_df["jo_pass"] == True) &
        (all_df["stab"]   >= STABILITY_MIN_PCT)
    )
    qual = all_df[mask].sort_values(["eg_p", "stab"], ascending=[True, False]).reset_index(drop=True)
    print(f"  Stage 3+4 (+ Stability + HL): {len(qual)} qualified pairs")
    return qual, all_df

# -----------------------------------------------------------------
# SECTION 3 - KALMAN FILTER + SIGNAL GENERATION
# -----------------------------------------------------------------

def kalman_filter(y_arr, x_arr, delta=KALMAN_DELTA, Ve=KALMAN_VE):
    """
    Dynamic hedge ratio via Kalman filter.
    State:  beta_t  (scalar)
    Obs:    y_t = beta_t * x_t + eps_t
    Trans:  beta_t = beta_{t-1} + w_t    (random walk prior)
    R_t updated adaptively with EWMA of squared innovations.
    """
    n    = len(y_arr)
    beta = np.zeros(n)
    P    = np.zeros(n)
    warm = min(30, n // 5)
    cov  = np.cov(y_arr[:warm], x_arr[:warm])
    beta[0] = cov[0, 1] / np.var(x_arr[:warm]) if np.var(x_arr[:warm]) > 0 else 1.0
    P[0] = 1.0
    R    = Ve

    for t in range(1, n):
        # predict
        bp  = beta[t-1]
        Pp  = P[t-1] + delta
        # observe
        xt  = x_arr[t]
        inn = y_arr[t] - bp * xt
        S   = Pp * xt**2 + R
        K   = Pp * xt / S
        beta[t] = bp + K * inn
        P[t]    = max((1 - K * xt) * Pp, 1e-12)
        R       = 0.95 * R + 0.05 * inn**2

    return beta, P

def make_signals(df, ta, tb, hl, use_kalman=True):
    y, x, idx = df[ta].values, df[tb].values, df.index

    if use_kalman:
        beta_arr, _ = kalman_filter(y, x)
    else:
        b = ols_beta(df, ta, tb)
        beta_arr = np.full(len(y), b)

    spr      = pd.Series(y - beta_arr * x, index=idx, name="spread")
    beta_ser = pd.Series(beta_arr, index=idx, name="beta")
    win      = max(10, int(round(hl * 0.75)))
    mu       = spr.rolling(win).mean()
    sd       = spr.rolling(win).std()
    zs       = ((spr - mu) / sd).rename("zscore")

    return spr, zs, beta_ser, win

# -----------------------------------------------------------------
# SECTION 4 - REGIME DETECTION
# -----------------------------------------------------------------

def vol_regime(df, ta, tb):
    """
    1 = high-vol regime (top REGIME_PCT_HIGH of rolling RV), 0 = normal.
    Computed on the OLS spread so it doesn't depend on Kalman state.
    """
    b    = ols_beta(df, ta, tb)
    spr  = df[ta] - b * df[tb]
    rv   = spr.pct_change().dropna().rolling(REGIME_WINDOW).std() * np.sqrt(252)
    thr  = rv.quantile(REGIME_PCT_HIGH)
    reg  = (rv >= thr).astype(int).reindex(df.index).fillna(0)
    return reg

# -----------------------------------------------------------------
# SECTION 5 - BACKTEST ENGINE
# -----------------------------------------------------------------

def trade_cost(pa, pb, beta, cap):
    sa = cap / pa
    sb = abs(cap * beta) / pb
    return COMM_PER_SHARE * (sa + sb) * 2 + (SLIP_BPS / 1e4) * (cap + abs(cap * beta)) * 2

def run_backtest(df, ta, tb, spr, zs, beta_ser, regime=None, cap=CAPITAL):
    pa  = df[ta].reindex(zs.dropna().index)
    pb  = df[tb].reindex(zs.dropna().index)
    zs  = zs.dropna()
    spr = spr.reindex(zs.index)
    bet = beta_ser.reindex(zs.index)
    reg = (regime.reindex(zs.index).fillna(0) if regime is not None
           else pd.Series(0, index=zs.index))

    pos, trades, entry = 0, [], {}

    for i in range(1, len(zs)):
        zp, zn = zs.iloc[i-1], zs.iloc[i]
        dt = zs.index[i]

        if pos == 0:
            if SKIP_HIGH_REGIME and reg.iloc[i] == 1:
                continue
            if zp > ENTRY_Z:
                pos = -1
                entry = {"dt": dt, "z": zp, "pa": pa.iloc[i],
                         "pb": pb.iloc[i], "beta": bet.iloc[i],
                         "spr": spr.iloc[i]}
            elif zp < -ENTRY_Z:
                pos = +1
                entry = {"dt": dt, "z": zp, "pa": pa.iloc[i],
                         "pb": pb.iloc[i], "beta": bet.iloc[i],
                         "spr": spr.iloc[i]}
        else:
            hit_exit = (pos == +1 and zn >= -EXIT_Z) or (pos == -1 and zn <= EXIT_Z)
            hit_stop = abs(zn) >= STOP_Z
            if hit_exit or hit_stop:
                ep, ebp = pa.iloc[i], pb.iloc[i]
                eb = entry["beta"]
                sa = cap / entry["pa"]
                sb = abs(cap * eb) / entry["pb"]
                gross = pos * sa * (ep - entry["pa"]) - pos * sb * (ebp - entry["pb"])
                cost  = trade_cost(entry["pa"], entry["pb"], eb, cap)
                net   = gross - cost

                mfe = pos * (spr.iloc[i] - entry["spr"])
                mae = -pos * (spr.iloc[i] - entry["spr"])

                trades.append({
                    "entry_date":  entry["dt"],
                    "exit_date":   dt,
                    "direction":   "long_a" if pos == 1 else "short_a",
                    "entry_z":     round(entry["z"], 3),
                    "exit_z":      round(zn, 3),
                    "beta":        round(eb, 4),
                    "hold_days":   (dt - entry["dt"]).days,
                    "gross_pnl":   round(gross, 2),
                    "cost":        round(cost, 2),
                    "net_pnl":     round(net, 2),
                    "mfe":         round(mfe, 4),
                    "mae":         round(mae, 4),
                    "exit_reason": "stop" if hit_stop else "reversion",
                    "regime":      int(reg.iloc[i]),
                })
                pos = 0

    if not trades:
        return {"n_trades": 0}

    tdf = pd.DataFrame(trades)
    return {
        "n_trades":   len(tdf),
        "trades_df":  tdf,
        "spread":     spr,
        "zscore":     zs,
        "beta_series":bet,
        "prices_a":   pa,
        "prices_b":   pb,
    }

# -----------------------------------------------------------------
# SECTION 6 - ANALYTICS ENGINE
# -----------------------------------------------------------------

def compute_analytics(tdf: pd.DataFrame, cap: float, label: str = "") -> dict:
    if tdf is None or tdf.empty:
        return {"label": label, "n_trades": 0}

    pnl  = tdf["net_pnl"].values
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd   = cum - peak
    max_dd = float(dd.min())

    # Drawdown duration
    in_dd, dd_start, dd_durs = False, 0, []
    for i, v in enumerate(dd):
        if v < 0 and not in_dd:
            in_dd, dd_start = True, i
        elif v >= 0 and in_dd:
            in_dd = False
            dd_durs.append(i - dd_start)
    max_dd_dur = max(dd_durs) if dd_durs else 0

    # Daily PnL for ratio calculations
    daily = tdf.set_index("exit_date")["net_pnl"].resample("D").sum().fillna(0)
    mu, sd = daily.mean(), daily.std()
    dd_sd  = daily[daily < 0].std() if (daily < 0).any() else 1e-9

    sharpe  = mu / sd * np.sqrt(252) if sd > 0 else 0.0
    sortino = mu / dd_sd * np.sqrt(252) if dd_sd > 0 else 0.0

    n_days  = max((tdf["exit_date"].max() - tdf["entry_date"].min()).days, 1)
    total   = float(cum[-1])
    cagr    = (1 + total / (2 * cap)) ** (365 / n_days) - 1
    calmar  = cagr / abs(max_dd / (2 * cap)) if max_dd < 0 else 0.0

    wins   = tdf[tdf["net_pnl"] > 0]["net_pnl"]
    losses = tdf[tdf["net_pnl"] < 0]["net_pnl"].abs()
    w_rate = len(wins) / len(tdf) if len(tdf) > 0 else 0
    avg_w  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_l  = float(losses.mean()) if len(losses) > 0 else 1e-9
    payoff = avg_w / avg_l if avg_l > 0 else 0.0
    pf     = wins.sum() / losses.sum() if losses.sum() > 0 else float("inf")

    kelly = 0.0
    if payoff > 0 and avg_l > 0:
        kelly = max(0.0, min((w_rate * payoff - (1 - w_rate)) / payoff, MAX_KELLY))
        if HALF_KELLY:
            kelly /= 2

    # VaR / CVaR (95%)
    sp = np.sort(daily.values)
    idx95 = max(int(0.05 * len(sp)), 1)
    var95  = float(-sp[idx95])
    cvar95 = float(-sp[:idx95].mean()) if idx95 > 0 else var95

    # Info ratio vs zero (alpha / tracking error)
    info_ratio = sharpe   # vs zero-return benchmark; same formula

    n_years  = max(n_days / 365, 1e-6)
    turnover = len(tdf) * 2 / n_years  # round-trips per year

    return {
        "label":         label,
        "n_trades":      len(tdf),
        "total_pnl":     round(total, 2),
        "cagr_%":        round(cagr * 100, 2),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "calmar":        round(calmar, 3),
        "info_ratio":    round(info_ratio, 3),
        "max_dd":        round(max_dd, 2),
        "max_dd_days":   max_dd_dur,
        "win_rate_%":    round(w_rate * 100, 1),
        "avg_win":       round(avg_w, 2),
        "avg_loss":      round(avg_l, 2),
        "payoff_ratio":  round(payoff, 3),
        "profit_factor": round(pf, 3),
        "kelly_f":       round(kelly, 4),
        "var_95_d":      round(var95, 2),
        "cvar_95_d":     round(cvar95, 2),
        "avg_hold_d":    round(tdf["hold_days"].mean(), 1),
        "stop_rate_%":   round((tdf["exit_reason"] == "stop").mean() * 100, 1),
        "total_cost":    round(tdf["cost"].sum(), 2),
        "turnover_pa":   round(turnover, 1),
    }

# -----------------------------------------------------------------
# SECTION 7 - MONTE CARLO SIGNIFICANCE
# -----------------------------------------------------------------

def mc_pvalue(tdf: pd.DataFrame, actual_sharpe: float, n=MC_ITERS) -> float:
    if tdf is None or len(tdf) < 5:
        return float("nan")
    pnl = tdf["net_pnl"].values
    avg_hold = max(tdf["hold_days"].mean(), 1)
    count = 0
    for _ in range(n):
        shuf = np.random.permutation(pnl)
        mu, sd = shuf.mean(), shuf.std()
        rand_s = mu / sd * np.sqrt(252 / avg_hold) if sd > 0 else 0
        if rand_s >= actual_sharpe:
            count += 1
    return count / n

# -----------------------------------------------------------------
# SECTION 8 - WALK-FORWARD
# -----------------------------------------------------------------

def _oos_window(train_df, test_df, ta, tb):
    """Fit on train (OLS, frozen), evaluate OOS. Returns analytics dict."""
    b     = ols_beta(train_df, ta, tb)
    spr_t = train_df[ta] - b * train_df[tb]
    hl    = half_life(spr_t)
    hl    = float(np.clip(hl, HALF_LIFE_MIN, HALF_LIFE_MAX))
    win   = max(10, int(round(hl * 0.75)))

    spr_oos = test_df[ta] - b * test_df[tb]
    mu      = spr_oos.rolling(win).mean()
    sd      = spr_oos.rolling(win).std()
    zs_oos  = (spr_oos - mu) / sd

    beta_frozen = pd.Series(b, index=test_df.index)
    eg_p = eg_test(train_df[ta].values, train_df[tb].values)
    adf_p = float(adfuller(spr_t.dropna())[1])

    res = run_backtest(test_df, ta, tb, spr_oos, zs_oos, beta_frozen)
    m   = compute_analytics(res.get("trades_df"), CAPITAL) if res.get("n_trades", 0) > 0 else {}

    return {
        "oos_trades": res.get("n_trades", 0),
        "oos_pnl":    m.get("total_pnl", 0),
        "oos_sharpe": m.get("sharpe", 0),
        "oos_win_%":  m.get("win_rate_%", 0),
        "oos_calmar": m.get("calmar", 0),
        "eg_p":       round(eg_p, 4),
        "adf_p":      round(adf_p, 4),
        "beta":       round(b, 4),
        "half_life":  round(hl, 1),
        "trades_df":  res.get("trades_df"),
    }

def wf_rolling(df, ta, tb):
    n, rows, all_t = len(df), [], []
    i, wnum = 0, 1
    while i + WF_TRAIN + WF_TEST <= n:
        tr  = df.iloc[i : i + WF_TRAIN]
        te  = df.iloc[i + WF_TRAIN : i + WF_TRAIN + WF_TEST]
        res = _oos_window(tr, te, ta, tb)
        row = {"wf_type": "rolling", "window": f"R{wnum}",
               "train_start": tr.index[0].date(), "train_end": tr.index[-1].date(),
               "oos_start": te.index[0].date(), "oos_end": te.index[-1].date(), **res}
        rows.append(row)
        if res["trades_df"] is not None and len(res["trades_df"]) > 0:
            t = res["trades_df"].copy()
            t["window"] = f"R{wnum}"
            all_t.append(t)
        i += WF_STEP
        wnum += 1
    wf = pd.DataFrame([{k: v for k, v in r.items() if k != "trades_df"} for r in rows])
    trades = pd.concat(all_t) if all_t else pd.DataFrame()
    return wf, trades

def wf_anchored(df, ta, tb):
    n, rows, all_t = len(df), [], []
    train_end, wnum = WF_TRAIN, 1
    while train_end + WF_TEST <= n:
        tr  = df.iloc[:train_end]
        te  = df.iloc[train_end : train_end + WF_TEST]
        res = _oos_window(tr, te, ta, tb)
        row = {"wf_type": "anchored", "window": f"A{wnum}",
               "train_bars": len(tr),
               "train_start": tr.index[0].date(), "train_end": tr.index[-1].date(),
               "oos_start": te.index[0].date(), "oos_end": te.index[-1].date(), **res}
        rows.append(row)
        if res["trades_df"] is not None and len(res["trades_df"]) > 0:
            t = res["trades_df"].copy()
            t["window"] = f"A{wnum}"
            all_t.append(t)
        train_end += WF_STEP
        wnum += 1
    wf = pd.DataFrame([{k: v for k, v in r.items() if k != "trades_df"} for r in rows])
    trades = pd.concat(all_t) if all_t else pd.DataFrame()
    return wf, trades

# -----------------------------------------------------------------
# SECTION 9 - PORTFOLIO CONSTRUCTION
# -----------------------------------------------------------------

def build_portfolio(pair_list, total_cap=PORT_CAPITAL):
    """
    Kelly-weighted, spread-correlation-penalized multi-pair portfolio.
    Returns merged trade log scaled to portfolio weights.
    """
    # Spread correlation matrix
    spreads = {}
    for p in pair_list:
        if "spread" in p and p["spread"] is not None:
            spreads[p["pair"]] = p["spread"]
    spr_df   = pd.DataFrame(spreads).dropna(how="all")
    corr_mat = spr_df.corr() if not spr_df.empty else pd.DataFrame()

    # Kelly fractions per pair
    kelly = {}
    for p in pair_list:
        tdf = p.get("trades_df")
        m   = compute_analytics(tdf, CAPITAL) if tdf is not None and len(tdf) > 0 else {}
        kelly[p["pair"]] = m.get("kelly_f", 0.0)

    # Correlation penalty
    weights = {}
    for p in pair_list:
        key = p["pair"]
        kf  = kelly.get(key, 0.0)
        pen = 1.0
        if not corr_mat.empty and key in corr_mat.columns:
            for other in corr_mat.columns:
                if other != key and other in kelly:
                    c = abs(corr_mat.loc[key, other])
                    if c > SPREAD_CORR_CUTOFF and kelly.get(other, 0) > kf:
                        pen = max(0.25, 1 - c)
        weights[key] = max(0.0, kf * pen)

    n_pairs = max(len(pair_list), 1)
    alloc   = total_cap / n_pairs

    all_trades = []
    for p in pair_list:
        key = p["pair"]
        tdf = p.get("trades_df")
        w   = weights.get(key, 0.0)
        if tdf is None or len(tdf) == 0 or w <= 0:
            continue
        scale = w * alloc / CAPITAL
        t = tdf.copy()
        t["net_pnl"]   *= scale
        t["gross_pnl"] *= scale
        t["cost"]      *= scale
        t["pair"]       = key
        all_trades.append(t)

    if not all_trades:
        return {}

    port_trades = pd.concat(all_trades).sort_values("exit_date").reset_index(drop=True)
    port_m      = compute_analytics(port_trades, total_cap, label="Portfolio")

    return {"trades": port_trades, "metrics": port_m,
            "weights": weights, "kelly": kelly, "corr": corr_mat}

# -----------------------------------------------------------------
# SECTION 10 - MARKET NEUTRALITY
# -----------------------------------------------------------------

def market_neutrality(port_trades, spy):
    daily_pnl = port_trades.set_index("exit_date")["net_pnl"].resample("D").sum()
    spy_ret   = spy.pct_change().dropna()
    idx = daily_pnl.index.intersection(spy_ret.index)
    if len(idx) < 30:
        return {}
    y = daily_pnl[idx].values
    X = add_constant(spy_ret[idx].values)
    res = OLS(y, X).fit()
    return {
        "daily_alpha": round(float(res.params[0]), 2),
        "mkt_beta":    round(float(res.params[1]), 4),
        "r2":          round(float(res.rsquared), 4),
        "beta_pval":   round(float(res.pvalues[1]), 4),
        "neutral":     abs(float(res.params[1])) < 0.05,
    }

# -----------------------------------------------------------------
# SECTION 11 - PLOTS
# -----------------------------------------------------------------

def savefig(name):
    p = f"{OUT}/{name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")

def plot_screening(qual, all_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Pair Screening Landscape", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(all_df["eg_p"], bins=25, color="steelblue", alpha=0.75)
    ax.axvline(EG_PVAL_THRESHOLD, color="red", linestyle="--",
               label=f"p={EG_PVAL_THRESHOLD}")
    ax.set_title("EG p-value Distribution (all pairs)")
    ax.set_xlabel("p-value"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    sc = ax.scatter(all_df["eg_p"], all_df["stab"],
                    c=all_df["hl"].clip(HALF_LIFE_MIN, HALF_LIFE_MAX),
                    cmap="viridis", alpha=0.6, s=25)
    plt.colorbar(sc, ax=ax, label="Half-life (d)")
    if not qual.empty:
        ax.scatter(qual["eg_p"], qual["stab"], color="red",
                   s=90, marker="*", zorder=5, label="Qualified")
    ax.axvline(EG_PVAL_THRESHOLD, color="red", linestyle="--", alpha=0.5)
    ax.axhline(STABILITY_MIN_PCT, color="orange", linestyle="--", alpha=0.5)
    ax.set_title("Stability vs EG p-value (colour = half-life)")
    ax.set_xlabel("EG p-value"); ax.set_ylabel("Stability")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if not qual.empty:
        labels = [f"{r.ta}/{r.tb}" for _, r in qual.iterrows()]
        ax.barh(labels, qual["hl"].values, color="darkorange", alpha=0.8)
        ax.axvline(HALF_LIFE_MIN, color="red", linestyle="--")
        ax.axvline(HALF_LIFE_MAX, color="red", linestyle="--")
        ax.set_title("Half-Life: Qualified Pairs (days)")
        ax.set_xlabel("Days"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    both = ((all_df["eg_p"] < EG_PVAL_THRESHOLD) & (all_df["jo_pass"])).sum()
    eg_only = ((all_df["eg_p"] < EG_PVAL_THRESHOLD) & (~all_df["jo_pass"])).sum()
    neither = len(all_df) - both - eg_only
    ax.pie([both, eg_only, max(neither, 0)],
           labels=["EG + Johansen", "EG only", "Neither"],
           colors=["#2ecc71", "#f39c12", "#bdc3c7"],
           autopct="%1.0f%%", startangle=90)
    ax.set_title("Test Agreement")

    plt.tight_layout()
    savefig("01_screening_landscape")

def plot_pair_dashboard(bt, m, ta, tb):
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"{ta}/{tb}  |  Sharpe={m['sharpe']:.3f}  Sortino={m['sortino']:.3f}  "
        f"Calmar={m['calmar']:.3f}  CAGR={m['cagr_%']:.1f}%  "
        f"Win={m['win_rate_%']:.1f}%  PF={m['profit_factor']:.2f}  "
        f"MC_p={m.get('mc_p', '?')}",
        fontsize=11, fontweight="bold"
    )

    tdf  = bt["trades_df"]
    spr  = bt["spread"]
    zs   = bt["zscore"]
    pa   = bt["prices_a"]
    pb   = bt["prices_b"]
    bet  = bt["beta_series"]
    cum  = tdf["net_pnl"].cumsum().reset_index(drop=True)
    peak = cum.cummax()

    # 1. Normalised prices
    ax = fig.add_subplot(gs[0, :2])
    (pa / pa.iloc[0]).plot(ax=ax, label=ta, color="steelblue", lw=0.9)
    (pb / pb.iloc[0]).plot(ax=ax, label=tb, color="darkorange", lw=0.9)
    ax.set_title("Normalised Prices"); ax.legend(); ax.grid(alpha=0.3)

    # 2. Kalman beta
    ax = fig.add_subplot(gs[0, 2])
    bet.plot(ax=ax, color="purple", lw=0.8)
    ax.set_title("Kalman Hedge Ratio beta(t)"); ax.grid(alpha=0.3)

    # 3. Spread
    ax = fig.add_subplot(gs[1, 0])
    spr.plot(ax=ax, color="teal", lw=0.7, alpha=0.85)
    spr.rolling(30).mean().plot(ax=ax, color="crimson", lw=1.2, label="30d MA")
    ax.set_title(f"Spread  (ADF p={m.get('adf_p','?')})"); ax.legend(); ax.grid(alpha=0.3)

    # 4. Z-score
    ax = fig.add_subplot(gs[1, 1])
    zs.plot(ax=ax, color="navy", lw=0.7, alpha=0.85)
    for lv, col, ls in [(ENTRY_Z,"red","--"),(-ENTRY_Z,"red","--"),
                        (EXIT_Z,"green",":")  ,(-EXIT_Z,"green",":"),
                        (STOP_Z,"orange","-.")  ,(-STOP_Z,"orange","-.")]:
        ax.axhline(lv, color=col, ls=ls, lw=0.8, alpha=0.7)
    ax.set_title("Z-Score + Levels"); ax.grid(alpha=0.3)

    # 5. Equity + drawdown shading
    ax = fig.add_subplot(gs[1, 2])
    cum.plot(ax=ax, color="darkgreen", lw=1.5)
    ax.fill_between(cum.index, cum, peak, alpha=0.2, color="red", label="DD")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Equity Curve  (${m['total_pnl']:,.0f} net)")
    ax.set_xlabel("Trade #"); ax.legend(); ax.grid(alpha=0.3)

    # 6. Holding period histogram
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(tdf["hold_days"], bins=20, color="steelblue", alpha=0.75)
    ax.axvline(tdf["hold_days"].mean(), color="red", ls="--",
               label=f"mu={m['avg_hold_d']:.1f}d")
    ax.set_title("Holding Period"); ax.set_xlabel("Days")
    ax.legend(); ax.grid(alpha=0.3)

    # 7. MAE vs MFE
    ax = fig.add_subplot(gs[2, 1])
    colors_t = ["green" if p > 0 else "red" for p in tdf["net_pnl"]]
    ax.scatter(tdf["mae"], tdf["mfe"], c=colors_t, alpha=0.6, s=35)
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set_title("MAE vs MFE (spread)"); ax.grid(alpha=0.3)
    ax.set_xlabel("MAE (adverse)"); ax.set_ylabel("MFE (favorable)")

    # 8. Per-trade PnL + rolling win rate
    ax = fig.add_subplot(gs[2, 2])
    bar_c = ["green" if p > 0 else "red" for p in tdf["net_pnl"]]
    ax.bar(range(len(tdf)), tdf["net_pnl"], color=bar_c, alpha=0.75)
    ax2 = ax.twinx()
    (tdf["net_pnl"] > 0).rolling(5).mean().mul(100).reset_index(drop=True).plot(
        ax=ax2, color="navy", lw=1.3, alpha=0.75)
    ax2.axhline(50, color="gray", ls="--", lw=0.7)
    ax2.set_ylabel("Roll.5 Win%", color="navy", fontsize=8)
    ax.set_title("Per-Trade PnL + Rolling Win Rate")
    ax.set_xlabel("Trade #"); ax.grid(alpha=0.3)

    plt.tight_layout()
    savefig(f"02_dashboard_{ta}_{tb}")

def plot_wf(wf_r, wf_a, oos_r, oos_a, ta, tb):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Walk-Forward: {ta}/{tb}  (Rolling vs Anchored)",
                 fontsize=13, fontweight="bold")

    for row_idx, (wf, oos, label, color) in enumerate([
        (wf_r, oos_r, "Rolling",  "steelblue"),
        (wf_a, oos_a, "Anchored", "darkorange"),
    ]):
        if wf.empty:
            continue

        ax = axes[row_idx, 0]
        if "oos_pnl" in wf.columns:
            bc = ["green" if p > 0 else "red" for p in wf["oos_pnl"].fillna(0)]
            ax.bar(wf["window"], wf["oos_pnl"].fillna(0), color=bc, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{label}: OOS PnL/Window"); ax.grid(alpha=0.3)

        ax = axes[row_idx, 1]
        if "oos_sharpe" in wf.columns:
            ax.plot(wf["window"], wf["oos_sharpe"].fillna(0),
                    marker="o", color=color, lw=1.5)
            ax.axhline(1.0, color="green", ls="--", lw=0.9, label="Sharpe=1")
            ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(f"{label}: OOS Sharpe"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[row_idx, 2]
        if not oos.empty:
            oos_cum = oos.sort_values("exit_date")["net_pnl"].cumsum().reset_index(drop=True)
            oos_cum.plot(ax=ax, color=color, lw=1.5)
            ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{label}: Aggregated OOS Equity")
        ax.set_xlabel("Trade #"); ax.grid(alpha=0.3)

    plt.tight_layout()
    savefig(f"03_wf_{ta}_{tb}")

def plot_portfolio(port):
    if not port or "trades" not in port:
        return
    trades  = port["trades"]
    metrics = port["metrics"]

    daily = trades.set_index("exit_date")["net_pnl"].resample("D").sum().fillna(0)
    cum   = daily.cumsum()
    peak  = cum.cummax()
    dd    = cum - peak

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Multi-Pair Portfolio  |  Sharpe={metrics['sharpe']:.3f}  "
        f"Sortino={metrics['sortino']:.3f}  CAGR={metrics['cagr_%']:.1f}%  "
        f"MaxDD=${metrics['max_dd']:,.0f}  Calmar={metrics['calmar']:.3f}",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0, 0]
    cum.plot(ax=ax, color="darkgreen", lw=1.5)
    ax.fill_between(cum.index, cum, peak, alpha=0.2, color="red", label="DD")
    ax.set_title("Portfolio Equity Curve")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    dd.plot(ax=ax, color="red", lw=1.2)
    ax.fill_between(dd.index, dd, 0, alpha=0.25, color="red")
    ax.set_title(f"Drawdown  (Max=${metrics['max_dd']:,.0f})"); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    roll_s = daily.rolling(60).mean() / daily.rolling(60).std() * np.sqrt(252)
    roll_s.plot(ax=ax, color="steelblue", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="green", ls="--", lw=0.8)
    ax.set_title("Rolling 60d Sharpe"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if "pair" in trades.columns:
        pp = trades.groupby("pair")["net_pnl"].sum().sort_values()
        bc = ["green" if v > 0 else "red" for v in pp]
        pp.plot(kind="barh", ax=ax, color=bc, alpha=0.8)
    ax.set_title("PnL by Pair"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    w = port.get("weights", {})
    if w:
        keys, vals = list(w.keys()), [w[k] for k in w]
        ax.bar(range(len(keys)), vals, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_title("Kelly-Adj Weights"); ax.set_ylabel("Weight"); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    try:
        monthly = daily.resample("ME").sum()
        mdf = monthly.to_frame("pnl")
        mdf["year"]  = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot(index="year", columns="month", values="pnl")
        pivot.columns = [["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"][c-1]
                         for c in pivot.columns]
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0,
                    annot=True, fmt=".0f", annot_kws={"size": 7}, linewidths=0.3)
        ax.set_title("Monthly PnL Heatmap ($)")
    except Exception:
        daily.resample("ME").sum().plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
        ax.set_title("Monthly PnL")

    plt.tight_layout()
    savefig("04_portfolio_dashboard")

# -----------------------------------------------------------------
# SECTION 12 - ORCHESTRATOR
# -----------------------------------------------------------------

def banner(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def print_table(rows, cols, title):
    df = pd.DataFrame(rows)[cols]
    pd.set_option("display.float_format", "{:,.3f}".format)
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.width", 200)
    print(f"\n  {title}:\n{df.to_string(index=False)}")

def run():
    banner("ADVANCED STAT-ARB RESEARCH FRAMEWORK")
    print(f"  Data={DATA_PERIOD} daily  "
          f"EG_thr={EG_PVAL_THRESHOLD}  "
          f"Stab_min={STABILITY_MIN_PCT}  "
          f"HL=[{HALF_LIFE_MIN},{HALF_LIFE_MAX}]d  "
          f"Kalman=ON  RegimeFilter={'ON' if SKIP_HIGH_REGIME else 'OFF'}")

    # -- 1. DATA ----------------------------------------------------------------
    banner("STEP 1 - DATA")
    universe = build_universe()
    print(f"  Universe: {len(universe)} tickers")
    prices = fetch_prices(universe)
    available = list(prices.columns)
    print(f"  Available after quality filter: {len(available)}")
    spy = prices["SPY"] if "SPY" in available else None

    # -- 2. SCREENING -----------------------------------------------------------
    banner("STEP 2 - COINTEGRATION BATTERY")
    qual, all_df = screen_pairs(prices)

    all_df.to_csv(f"{OUT}/all_pairs_screened.csv", index=False)
    if not qual.empty:
        qual.to_csv(f"{OUT}/qualified_pairs.csv", index=False)
        print(f"\n  Qualified pairs:\n{qual.to_string(index=False)}")
    plot_screening(qual, all_df)

    if qual.empty:
        print("\n  No pairs survived full battery. Relaxing to EG+Johansen only.")
        qual = all_df[(all_df["eg_p"] < EG_PVAL_THRESHOLD) &
                      (all_df["jo_pass"] == True)].head(8)

    # -- 3. FULL BACKTEST -------------------------------------------------------
    banner("STEP 3 - FULL BACKTEST (Kalman + Regime)")
    pair_results, metrics_list = [], []

    for _, row in qual.head(8).iterrows():
        ta, tb, hl = row.ta, row.tb, row.hl
        print(f"\n  -> {ta}/{tb}  HL={hl:.1f}d  EG_p={row.eg_p:.4f}  Stab={row.stab:.2f}")

        df_pair = prices[[ta, tb]].dropna()
        spr, zs, bet, win = make_signals(df_pair, ta, tb, hl, use_kalman=True)
        reg = vol_regime(df_pair, ta, tb)

        bt = run_backtest(df_pair, ta, tb, spr, zs, bet, regime=reg)
        if bt.get("n_trades", 0) == 0:
            print(f"     No trades generated.")
            continue

        m = compute_analytics(bt["trades_df"], CAPITAL, label=f"{ta}/{tb}")
        m["adf_p"]  = round(float(adfuller(spr.dropna())[1]), 4)
        m["hl"]     = hl
        m["zs_win"] = win
        m["mc_p"]   = round(mc_pvalue(bt["trades_df"], m["sharpe"]), 3)

        print(f"     n={m['n_trades']}  Sharpe={m['sharpe']:.3f}  "
              f"Sortino={m['sortino']:.3f}  CAGR={m['cagr_%']:.1f}%  "
              f"PF={m['profit_factor']:.2f}  WinR={m['win_rate_%']:.1f}%  "
              f"MC_p={m['mc_p']}  Kelly={m['kelly_f']:.3f}")

        metrics_list.append(m)
        pair_results.append({
            "pair": f"{ta}/{tb}", "ta": ta, "tb": tb,
            **m, **bt,
        })

        plot_pair_dashboard(bt, m, ta, tb)

    if metrics_list:
        key_cols = ["label","n_trades","total_pnl","cagr_%","sharpe","sortino",
                    "calmar","max_dd","win_rate_%","profit_factor","kelly_f",
                    "var_95_d","avg_hold_d","stop_rate_%","mc_p"]
        print_table(metrics_list,
                    [c for c in key_cols if c in metrics_list[0]],
                    "Full Backtest Results")

    # -- 4. WALK-FORWARD --------------------------------------------------------
    if pair_results:
        banner("STEP 4 - WALK-FORWARD (Rolling + Anchored)")
        best = max(metrics_list, key=lambda x: x.get("sharpe", 0))
        ta_b, tb_b = best["label"].split("/")
        df_wf = prices[[ta_b, tb_b]].dropna()
        print(f"  Best pair for WF: {ta_b}/{tb_b}  Sharpe={best['sharpe']:.3f}")
        print(f"  Train={WF_TRAIN}d  Test={WF_TEST}d  Step={WF_STEP}d")

        wf_r, oos_r = wf_rolling(df_wf, ta_b, tb_b)
        wf_a, oos_a = wf_anchored(df_wf, ta_b, tb_b)

        for tag, wf, oos in [("Rolling", wf_r, oos_r), ("Anchored", wf_a, oos_a)]:
            print(f"\n  {tag} ({len(wf)} windows):")
            dcols = [c for c in ["window","oos_start","oos_end","oos_trades",
                                  "oos_pnl","oos_sharpe","oos_win_%","eg_p"] if c in wf.columns]
            if not wf.empty:
                print(wf[dcols].to_string(index=False))
            if not oos.empty:
                agg_pnl = oos["net_pnl"].sum()
                agg_wr  = (oos["net_pnl"] > 0).mean() * 100
                n_pos   = (wf["oos_pnl"] > 0).sum() if "oos_pnl" in wf.columns else "?"
                print(f"  -> Agg OOS PnL=${agg_pnl:,.2f}  WinR={agg_wr:.1f}%  "
                      f"Prof.Windows={n_pos}/{len(wf)}")

        wf_r.to_csv(f"{OUT}/wf_rolling_{ta_b}_{tb_b}.csv", index=False)
        wf_a.to_csv(f"{OUT}/wf_anchored_{ta_b}_{tb_b}.csv", index=False)
        plot_wf(wf_r, wf_a, oos_r, oos_a, ta_b, tb_b)

    # -- 5. PORTFOLIO -----------------------------------------------------------
    if len(pair_results) > 1:
        banner("STEP 5 - MULTI-PAIR KELLY PORTFOLIO")
        port = build_portfolio(pair_results, PORT_CAPITAL)
        if port and "metrics" in port:
            pm = port["metrics"]
            print(f"\n  Portfolio ({len(pair_results)} pairs, "
                  f"capital=${PORT_CAPITAL:,.0f}):")
            for k, v in pm.items():
                if k != "label":
                    print(f"    {k:<22} {v}")

            if spy is not None and not port["trades"].empty:
                banner("STEP 5b - MARKET NEUTRALITY")
                mn = market_neutrality(port["trades"], spy)
                verdict = "NEUTRAL" if mn.get("neutral") else "EXPOSED"
                print(f"  Daily alpha:   ${mn.get('daily_alpha', 0):,.2f}")
                print(f"  Market beta:    {mn.get('mkt_beta', 0):.4f}  "
                      f"(p={mn.get('beta_pval', 0):.4f})")
                print(f"  R2:             {mn.get('r2', 0):.4f}")
                print(f"  Verdict:        {verdict} "
                      f"({'|beta|<0.05' if mn.get('neutral') else '|beta|>=0.05'})")

            plot_portfolio(port)

    # -- 6. FINAL SUMMARY -------------------------------------------------------
    banner("FINAL SUMMARY")
    print(f"  Universe screened:    {len(universe)} tickers")
    print(f"  Pairs tested:         {len(all_df)}")
    print(f"  Pairs qualified:      {len(qual)}")
    print(f"  Pairs with trades:    {len(pair_results)}")

    if metrics_list:
        best_s = max(metrics_list, key=lambda x: x.get("sharpe", 0))
        best_c = max(metrics_list, key=lambda x: x.get("cagr_%", 0))
        best_p = max(metrics_list, key=lambda x: x.get("profit_factor", 0))
        print(f"\n  Best Sharpe:  {best_s['label']}  -> {best_s['sharpe']:.3f}")
        print(f"  Best CAGR:    {best_c['label']}  -> {best_c['cagr_%']:.1f}%")
        print(f"  Best PF:      {best_p['label']}  -> {best_p['profit_factor']:.2f}")

    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".png", ".csv")):
            print(f"    {f}")

    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return pair_results, qual, metrics_list

if __name__ == "__main__":
    run()
