"""
A-Share Sector Rotation Momentum Strategy
==========================================
Universe  : 28 Shenwan Level-1 industry indices
Signal    : Composite momentum — weighted average of 1m/3m/6m/12m returns
Selection : Top-N sectors (default N=5) rebalanced monthly
Costs     : 0.3% one-way (0.6% round-trip)
Benchmark : CSI 300 (sh000300)
Period    : 2005-01-01 → present
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import akshare as ak

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────

# Shenwan Level-1 industry index codes + names
SW_INDUSTRIES = {
    '801010': 'Agriculture',
    '801020': 'Mining',
    '801030': 'Chemicals',
    '801040': 'Steel',
    '801050': 'Non-Ferrous Metals',
    '801080': 'Electronics',
    '801110': 'Home Appliances',
    '801120': 'Food & Beverage',
    '801130': 'Textiles & Apparel',
    '801140': 'Light Manufacturing',
    '801150': 'Pharma & Biotech',
    '801160': 'Utilities',
    '801170': 'Transportation',
    '801180': 'Real Estate',
    '801200': 'Commerce & Trade',
    '801210': 'Leisure Services',
    '801230': 'Conglomerates',
    '801710': 'Construction Materials',
    '801720': 'Construction Decoration',
    '801730': 'Electrical Equipment',
    '801740': 'Defense',
    '801750': 'IT & Computer',
    '801760': 'Media',
    '801770': 'Telecom',
    '801780': 'Banking',
    '801790': 'Non-Bank Finance',
    '801880': 'Automotive',
    '801890': 'Machinery',
}

TOP_N = 3                  # number of sectors to hold (concentrated)
LOOKBACK_MONTHS = [1, 3, 6, 12]  # momentum windows
MOM_WEIGHTS    = [0.4, 0.3, 0.2, 0.1]  # weights for each window (sum=1)
TC_ONE_WAY     = 0.003     # transaction cost per trade (0.3%)
START_DATE     = '20050101'
TREND_FILTER_WINDOW = 10   # months: only invest when benchmark > N-month MA
RISK_ADJ_MOM   = True      # divide momentum by trailing volatility (risk-adjusted)
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))

# ─── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_sw_monthly(code: str, retries: int = 3) -> pd.Series:
    """Fetch SW industry monthly close prices as a Series indexed by date."""
    for attempt in range(retries):
        try:
            df = ak.index_hist_sw(symbol=code, period='month')
            df = df[['日期', '收盘']].copy()
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期').sort_index()
            # keep month-end only (already monthly)
            return df['收盘'].rename(code)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  Failed {code}: {e}")
                return pd.Series(dtype=float, name=code)


def fetch_benchmark() -> pd.Series:
    """Fetch CSI 300 monthly close prices."""
    df = ak.stock_zh_index_daily(symbol='sh000300')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    monthly = df['close'].resample('ME').last()
    monthly.index = monthly.index.to_period('M').to_timestamp('M')
    return monthly.rename('CSI300')


def load_all_data() -> tuple[pd.DataFrame, pd.Series]:
    """Download all sector data and benchmark. Returns (prices_df, benchmark)."""
    cache_path = os.path.join(OUT_DIR, 'prices_cache.csv')

    if os.path.exists(cache_path):
        print("Loading from cache...")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {len(SW_INDUSTRIES)} industry indices...")
        series_list = []
        for i, (code, name) in enumerate(SW_INDUSTRIES.items(), 1):
            print(f"  [{i}/{len(SW_INDUSTRIES)}] {code} {name}")
            s = fetch_sw_monthly(code)
            if not s.empty:
                series_list.append(s)
            time.sleep(0.3)  # be polite
        prices = pd.concat(series_list, axis=1)
        prices.index = pd.to_datetime(prices.index)
        prices.to_csv(cache_path)
        print("Saved to cache.")

    print("Fetching benchmark (CSI 300)...")
    benchmark = fetch_benchmark()
    return prices, benchmark


# ─── Strategy Logic ────────────────────────────────────────────────────────────

def compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each month, compute composite momentum score for each sector.
    If RISK_ADJ_MOM is True, divide score by trailing 12m volatility.
    Returns a DataFrame of scores with same index/columns as prices.
    """
    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    monthly_rets = prices.pct_change()

    for t in range(max(LOOKBACK_MONTHS) + 1, len(prices)):
        row_scores = {}
        for col in prices.columns:
            s = 0.0
            for lb, w in zip(LOOKBACK_MONTHS, MOM_WEIGHTS):
                start = t - lb
                if start < 0:
                    continue
                r = (prices.iloc[t][col] / prices.iloc[start][col]) - 1
                if not np.isnan(r):
                    s += w * r
            # Risk-adjust: divide by trailing 12m volatility
            if RISK_ADJ_MOM:
                vol = monthly_rets.iloc[max(0, t-12):t][col].std()
                if vol > 0 and not np.isnan(vol):
                    s = s / vol
            row_scores[col] = s
        scores.iloc[t] = pd.Series(row_scores)

    return scores


def compute_trend_filter(benchmark: pd.Series, prices_index: pd.Index) -> pd.Series:
    """
    Return a boolean Series aligned to prices_index:
    True  = market uptrend (benchmark > N-month MA) → invest
    False = downtrend → hold cash
    """
    # Reindex benchmark to match prices dates (forward-fill gaps)
    bm = benchmark.reindex(prices_index, method='ffill')
    ma = bm.rolling(TREND_FILTER_WINDOW).mean()
    return (bm > ma).shift(1)  # use previous month's signal


def backtest(prices: pd.DataFrame, scores: pd.DataFrame,
             trend_filter: pd.Series = None) -> pd.DataFrame:
    """
    Simulate monthly rebalancing: hold top-N sectors by composite momentum.
    When trend_filter is provided and False → hold cash (0% return).
    Returns a DataFrame with columns: [strategy_return, turnover, in_market].
    """
    results = []
    held = set()

    for t in range(max(LOOKBACK_MONTHS) + 2, len(prices)):
        date = prices.index[t]
        prev_date = prices.index[t - 1]

        # Market timing: if trend filter says bear market → cash
        if trend_filter is not None:
            in_market = bool(trend_filter.iloc[t])
        else:
            in_market = True

        if not in_market:
            # Exit all positions
            if held:
                turnover = 1.0
                held = set()
            else:
                turnover = 0.0
            results.append({'date': date, 'strategy_return': 0.0,
                             'turnover': turnover, 'in_market': False})
            continue

        # Rank sectors by score at end of previous month
        row = scores.iloc[t - 1].dropna()
        if len(row) < TOP_N:
            results.append({'date': date, 'strategy_return': 0.0,
                             'turnover': 0.0, 'in_market': True})
            continue

        new_held = set(row.nlargest(TOP_N).index)

        # Turnover = fraction of portfolio replaced
        sells = held - new_held
        turnover = len(sells) / TOP_N if held else 1.0

        # Equal-weight return of top-N sectors from t-1 to t
        port_rets = []
        for col in new_held:
            p0 = prices.loc[prev_date, col]
            p1 = prices.loc[date, col]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                port_rets.append((p1 / p0) - 1)

        gross_ret = np.mean(port_rets) if port_rets else 0.0
        net_ret   = gross_ret - turnover * TC_ONE_WAY * 2  # buys + sells

        results.append({'date': date, 'strategy_return': net_ret,
                        'turnover': turnover, 'in_market': True})
        held = new_held

    return pd.DataFrame(results).set_index('date')


def align_benchmark(benchmark: pd.Series, strategy: pd.DataFrame) -> pd.Series:
    """Align benchmark returns to strategy dates."""
    bm_ret = benchmark.pct_change().reindex(strategy.index)
    return bm_ret.fillna(0)


# ─── Performance Metrics ───────────────────────────────────────────────────────

def calc_metrics(returns: pd.Series, label: str = '') -> dict:
    """Compute annualised performance metrics from monthly returns series."""
    r = returns.dropna()
    n = len(r)
    cumulative = (1 + r).cumprod()
    total_return = cumulative.iloc[-1] - 1
    years = n / 12
    cagr = (1 + total_return) ** (1 / years) - 1

    # Annualised volatility
    vol = r.std() * np.sqrt(12)

    # Sharpe (rf = 0 for simplicity; A-share risk-free ~2-3%)
    rf_monthly = 0.025 / 12
    sharpe = (r.mean() - rf_monthly) / r.std() * np.sqrt(12) if r.std() > 0 else 0

    # Max drawdown
    roll_max = cumulative.cummax()
    drawdown = (cumulative - roll_max) / roll_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    # Win rate
    win_rate = (r > 0).mean()

    return {
        'Label': label,
        'CAGR': f'{cagr:.1%}',
        'Total Return': f'{total_return:.1%}',
        'Ann. Volatility': f'{vol:.1%}',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_dd:.1%}',
        'Calmar Ratio': f'{calmar:.2f}',
        'Win Rate': f'{win_rate:.1%}',
        'Months': n,
    }


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(strategy_rets: pd.Series, bm_rets: pd.Series,
                 turnover: pd.Series, prices: pd.DataFrame, scores: pd.DataFrame,
                 base_rets: pd.Series = None):

    strat_cum = (1 + strategy_rets).cumprod()
    bm_cum    = (1 + bm_rets).cumprod()

    # Drawdown
    strat_dd = (strat_cum - strat_cum.cummax()) / strat_cum.cummax()
    bm_dd    = (bm_cum   - bm_cum.cummax())    / bm_cum.cummax()

    # Sector allocation heatmap (last 36 months)
    n_heatmap = min(36, len(scores))
    recent_scores = scores.iloc[-n_heatmap:]
    holdings = pd.DataFrame(0, index=recent_scores.index, columns=recent_scores.columns)
    for t in range(len(recent_scores)):
        row = recent_scores.iloc[t].dropna()
        if len(row) >= TOP_N:
            top = row.nlargest(TOP_N).index
            holdings.iloc[t][top] = 1

    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
    fig.suptitle('A-Share Sector Rotation Momentum Strategy\n'
                 f'Top-{TOP_N} Sectors | Monthly Rebalance | 0.3% Transaction Cost',
                 fontsize=16, color='white', y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.07, right=0.97, top=0.92, bottom=0.05)

    dark_bg = '#161b22'
    grid_color = '#30363d'
    text_color = '#c9d1d9'

    def style_ax(ax):
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)

    # ── Plot 1: Cumulative Returns ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(strat_cum.index, strat_cum.values, color='#3fb950', lw=2.5, label=f'Enhanced (Top-{TOP_N} + Trend Filter)')
    if base_rets is not None:
        base_cum = (1 + base_rets.reindex(strat_cum.index, fill_value=0)).cumprod()
        ax1.plot(base_cum.index, base_cum.values, color='#58a6ff', lw=1.5, alpha=0.8, label=f'Base Momentum (Top-{TOP_N})')
    ax1.plot(bm_cum.index,    bm_cum.values,    color='#f78166', lw=1.5, alpha=0.8, label='CSI 300')
    ax1.set_title('Cumulative Performance (log scale, net of 0.3% transaction costs)')
    ax1.set_yscale('log')
    ax1.legend(facecolor=dark_bg, edgecolor=grid_color, labelcolor=text_color, fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}x'))
    style_ax(ax1)

    # ── Plot 2: Drawdown ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(strat_dd.index, strat_dd.values, 0, alpha=0.7, color='#58a6ff', label='Strategy')
    ax2.fill_between(bm_dd.index,    bm_dd.values,    0, alpha=0.4, color='#f78166', label='CSI 300')
    ax2.set_title('Drawdown')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax2.legend(facecolor=dark_bg, edgecolor=grid_color, labelcolor=text_color, fontsize=9)
    style_ax(ax2)

    # ── Plot 3: Rolling 12m Return ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    roll12_strat = (1 + strategy_rets).rolling(12).apply(np.prod) - 1
    roll12_bm    = (1 + bm_rets).rolling(12).apply(np.prod) - 1
    ax3.plot(roll12_strat.index, roll12_strat.values * 100, color='#58a6ff', lw=1.5, label='Strategy')
    ax3.plot(roll12_bm.index,    roll12_bm.values    * 100, color='#f78166', lw=1.2, alpha=0.8, label='CSI 300')
    ax3.axhline(0, color=grid_color, lw=0.8)
    ax3.set_title('Rolling 12-Month Return (%)')
    ax3.legend(facecolor=dark_bg, edgecolor=grid_color, labelcolor=text_color, fontsize=9)
    style_ax(ax3)

    # ── Plot 4: Sector Allocation Heatmap ───────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    names = [SW_INDUSTRIES.get(c, c) for c in holdings.columns]
    im = ax4.imshow(holdings.T.values, aspect='auto', cmap='Blues',
                    vmin=0, vmax=1, interpolation='nearest')
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names, fontsize=7)
    # X-axis: show year labels
    x_dates = holdings.index
    year_ticks = [i for i, d in enumerate(x_dates) if d.month == 1]
    ax4.set_xticks(year_ticks)
    ax4.set_xticklabels([x_dates[i].year for i in year_ticks], fontsize=8)
    ax4.set_title(f'Sector Holdings — Last {n_heatmap} Months (Blue = Held)')
    ax4.set_facecolor(dark_bg)
    ax4.tick_params(colors=text_color, labelsize=7)
    ax4.title.set_color(text_color)
    for spine in ax4.spines.values():
        spine.set_color(grid_color)

    out = os.path.join(OUT_DIR, 'strategy_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Chart saved → {out}")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("A-Share Sector Rotation Momentum Strategy")
    print("=" * 60)

    # 1. Load data
    prices, benchmark = load_all_data()
    print(f"\nData loaded: {prices.shape[1]} sectors, "
          f"{prices.index[0].date()} → {prices.index[-1].date()}")

    # Filter to START_DATE
    prices    = prices[prices.index >= START_DATE]
    benchmark = benchmark[benchmark.index >= START_DATE]

    # Forward-fill missing values (holidays, new listings)
    prices = prices.ffill().dropna(how='all', axis=1)
    print(f"After filtering: {prices.shape[1]} sectors, {len(prices)} months")

    # 2. Compute momentum scores (risk-adjusted)
    print("\nComputing risk-adjusted momentum scores...")
    scores = compute_momentum(prices)

    # 3. Trend filter (market timing)
    print("Computing trend filter (CSI300 vs 10-month MA)...")
    trend = compute_trend_filter(benchmark, prices.index)

    # 4. Backtest — base (no filter) and enhanced (with filter)
    print("Running backtests...")
    bt_base     = backtest(prices, scores, trend_filter=None)
    bt_enhanced = backtest(prices, scores, trend_filter=trend)

    # 5. Align returns
    bm_rets        = align_benchmark(benchmark, bt_base)
    base_rets      = bt_base['strategy_return']
    enhanced_rets  = bt_enhanced['strategy_return']
    bm_rets_enh    = align_benchmark(benchmark, bt_enhanced)

    # 6. Metrics table
    base_m = calc_metrics(base_rets,     f'Momentum Top-{TOP_N}')
    enh_m  = calc_metrics(enhanced_rets, f'Enhanced (Trend Filter)')
    bm_m   = calc_metrics(bm_rets,       'CSI 300')

    print("\n" + "=" * 70)
    print(f"{'Metric':<22} {'Base Strategy':>14} {'Enhanced':>14} {'CSI 300':>14}")
    print("-" * 70)
    keys = ['CAGR', 'Total Return', 'Ann. Volatility', 'Sharpe Ratio',
            'Max Drawdown', 'Calmar Ratio', 'Win Rate', 'Months']
    for k in keys:
        print(f"{k:<22} {base_m[k]:>14} {enh_m[k]:>14} {bm_m[k]:>14}")
    print("=" * 70)

    # 7. Plot — use enhanced as primary
    print("\nGenerating charts...")
    plot_results(enhanced_rets, bm_rets_enh, bt_enhanced['turnover'], prices, scores,
                 base_rets=base_rets)

    # 8. Save monthly returns CSV
    out_csv = os.path.join(OUT_DIR, 'monthly_returns.csv')
    pd.DataFrame({
        'base_strategy':     base_rets,
        'enhanced_strategy': enhanced_rets,
        'csi300':            bm_rets,
        'turnover':          bt_enhanced['turnover'],
        'in_market':         bt_enhanced['in_market'],
    }).to_csv(out_csv)
    print(f"Monthly returns saved → {out_csv}")


if __name__ == '__main__':
    main()
