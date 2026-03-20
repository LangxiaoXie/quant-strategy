# A-Share Sector Rotation Strategy

A quantitative momentum strategy for Chinese A-shares, backtested from 2006–present with live data via akshare.

## Strategy

- **Universe**: 28 Shenwan Level-1 industry indices
- **Signal**: Risk-adjusted composite momentum (1m×0.4 + 3m×0.3 + 6m×0.2 + 12m×0.1, divided by trailing volatility)
- **Selection**: Top-3 sectors by score, equal-weighted
- **Rebalancing**: Monthly
- **Market timing**: Exit to cash when CSI 300 < 10-month moving average
- **Transaction costs**: 0.3% one-way (0.6% round-trip)

## Results vs 沪深300 (CSI 300) — 2006–2026

| Metric | Strategy | CSI 300 |
|---|---|---|
| **CAGR** | **13.9%** | 8.2% |
| **Total Return** | **1,277%** | 389% |
| Sharpe Ratio | 0.55 | 0.35 |
| Max Drawdown | -45% | -70.2% |
| Alpha/yr | **+8.9%** | — |
| Beta | 0.67 | 1.00 |

## Live Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard (interactive, live data) |
| `strategy.py` | Backtest engine + matplotlib charts |
| `compare.py` | Detailed comparison vs CSI 300 |
| `requirements.txt` | Python dependencies |
| `prices_cache.csv` | Cached monthly sector prices |
| `monthly_returns.csv` | Backtest monthly return series |

## Data Source

Shenwan Level-1 industry indices via [akshare](https://github.com/akfamily/akshare). CSI 300 benchmark from East Money.

## Disclaimer

Past performance does not guarantee future results. For educational/research purposes only.
