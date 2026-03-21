"""
JoinQuant-inspired A-Share Strategies
======================================
Strategy 1: Multi-Index Momentum Rotation
  - Universe: CSI300, CSI500, ChiNext + Bond proxy (cash)
  - Signal:   Risk-adjusted composite momentum (1m/3m/6m)
  - Timing:   Hold cash when ALL equity indices below MA
  - Rebal:    Monthly

Strategy 2: Small-Cap Premium + Quality Tilt (individual stocks)
  - Universe: 300 liquid small/mid-cap A-shares (fixed pool)
  - Signal:   Monthly price momentum (3m) × low volatility factor
  - Select:   Top 10 stocks, equal-weight
  - Timing:   Cash when CSI500 < 10m MA
  - Rebal:    Monthly
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import akshare as ak

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RF = 0.025 / 12  # monthly risk-free rate

# ─── Index codes ──────────────────────────────────────────────────────────────
INDICES = {
    'CSI300':  'sh000300',
    'CSI500':  'sh000905',
    'ChiNext': 'sz399006',
}
BENCHMARK = 'sh000300'

# ─── Helpers ──────────────────────────────────────────────────────────────────
def monthly_from_daily(sym: str) -> pd.Series:
    df = ak.stock_zh_index_daily(symbol=sym)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df['close'].resample('ME').last()


def drawdown(rets: pd.Series) -> pd.Series:
    cum = (1 + rets).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def metrics(rets: pd.Series, label='') -> dict:
    n   = len(rets)
    cum = (1 + rets).cumprod()
    tot = cum.iloc[-1] - 1
    cagr = (1 + tot) ** (12 / n) - 1
    vol  = rets.std() * np.sqrt(12)
    sh   = (rets.mean() - RF) / rets.std() * np.sqrt(12) if rets.std() > 0 else 0
    mdd  = drawdown(rets).min()
    cal  = cagr / abs(mdd) if mdd != 0 else 0
    wr   = (rets > 0).mean()
    return dict(label=label, cagr=cagr, tot=tot, vol=vol,
                sh=sh, mdd=mdd, cal=cal, wr=wr, n=n,
                cum=(1 + rets).cumprod(), dd=drawdown(rets))


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — Multi-Index Momentum Rotation
# ══════════════════════════════════════════════════════════════════════════════
def strategy1():
    print("=" * 60)
    print("Strategy 1: Multi-Index Momentum Rotation")
    print("=" * 60)

    # Load all index prices
    print("Loading index data...")
    prices = {}
    for name, sym in INDICES.items():
        prices[name] = monthly_from_daily(sym)
        time.sleep(0.3)
    bm = monthly_from_daily(BENCHMARK)
    time.sleep(0.3)

    df = pd.concat(prices, axis=1).dropna()

    # Momentum score: composite of 1m / 3m / 6m return
    def mom_score(col, t):
        s = 0.0
        for lb, w in [(1, 0.4), (3, 0.35), (6, 0.25)]:
            if t - lb >= 0:
                r = (df[col].iloc[t] / df[col].iloc[t - lb]) - 1
                s += w * r
        # risk-adjust by 6m vol
        vol = df[col].pct_change().iloc[max(0, t-6):t].std()
        return s / vol if vol > 0 else s

    # Market timing: cash if all indices below their 10m MA
    MA_WIN = 10

    results = []
    for t in range(MA_WIN + 1, len(df)):
        date = df.index[t]

        # Trend filter: at least 1 index above its MA → stay invested
        in_mkt = any(
            df[col].iloc[t - 1] > df[col].iloc[max(0, t - MA_WIN - 1):t - 1].mean()
            for col in df.columns
        )

        if not in_mkt:
            results.append({'date': date, 'ret': 0.0, 'held': 'Cash'})
            continue

        # Pick index with best momentum score (previous month's score)
        scores = {col: mom_score(col, t - 1) for col in df.columns}
        best   = max(scores, key=scores.get)

        # Return of the chosen index this month
        p0 = df[best].iloc[t - 1]
        p1 = df[best].iloc[t]
        ret = (p1 / p0) - 1 if p0 > 0 else 0.0
        results.append({'date': date, 'ret': ret, 'held': best})

    bt1 = pd.DataFrame(results).set_index('date')
    s1  = bt1['ret']
    bm1 = bm.pct_change().reindex(s1.index).fillna(0)

    m1   = metrics(s1,  "Index Rotation")
    mbm1 = metrics(bm1, "CSI 300")

    # Alpha / Beta
    cov  = np.cov(s1, bm1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
    alpha_ann = (s1.mean() - beta * bm1.mean()) * 12

    print(f"\n{'Metric':<22} {'Index Rotation':>16} {'CSI 300':>12}")
    print("-" * 52)
    for k, fmt in [('cagr',':.1%'),('tot',':.0%'),('vol',':.1%'),
                   ('sh',':.2f'),('mdd',':.1%'),('cal',':.2f'),('wr',':.1%')]:
        label = {'cagr':'CAGR','tot':'Total Return','vol':'Volatility',
                 'sh':'Sharpe','mdd':'Max Drawdown','cal':'Calmar','wr':'Win Rate'}[k]
        print(f"{label:<22} {format(m1[k], fmt[1:]):>16} {format(mbm1[k], fmt[1:]):>12}")
    print(f"{'Alpha/yr':<22} {alpha_ann:>+15.1%} {'—':>12}")
    print(f"{'Beta':<22} {beta:>16.2f} {'1.00':>12}")

    # Allocation history
    alloc = bt1['held'].value_counts()
    print(f"\nAllocation breakdown: {dict(alloc)}")

    return s1, bm1, m1, mbm1, bt1


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2 — Dual-Momentum Sector Rotation with Defensive Switch
# ══════════════════════════════════════════════════════════════════════════════
# Based on JoinQuant community 行业轮动 approach:
# - Dual momentum: hold a sector only if it has BOTH positive absolute return
#   AND top relative rank (avoids chasing sectors in secular downtrends)
# - Defensive rotation: when no sector qualifies, switch to the 3 least-volatile
#   defensive sectors (Banking, Utilities, Food & Beverage)
# - Momentum signal: 1m×0.5 + 3m×0.3 + 6m×0.2 (heavier on short-term)
# - Rebalance monthly, 0.3% one-way cost

DEFENSIVE = ['Banking', 'Utilities', 'Food & Beverage']
TOP_K     = 3
ABS_MOM_THRESHOLD = 0.0
TC        = 0.003


def strategy2():
    print("\n" + "=" * 60)
    print("Strategy 2: Dual-Momentum Sector Rotation (JoinQuant 行业轮动)")
    print("=" * 60)

    # Load SW sector prices (already cached)
    cache = os.path.join(OUT_DIR, 'prices_cache.csv')
    print("Loading SW sector prices from cache...")
    prices = pd.read_csv(cache, index_col=0, parse_dates=True).ffill()
    # Rename columns from codes to names
    code_to_name = {
        '801010': 'Agriculture',    '801020': 'Mining',
        '801030': 'Chemicals',      '801040': 'Steel',
        '801050': 'Non-Ferrous',    '801080': 'Electronics',
        '801110': 'Home Appliances','801120': 'Food & Beverage',
        '801130': 'Textiles',       '801140': 'Light Mfg',
        '801150': 'Pharma',         '801160': 'Utilities',
        '801170': 'Transport',      '801180': 'Real Estate',
        '801200': 'Commerce',       '801210': 'Leisure',
        '801230': 'Conglomerates',  '801710': 'Construction Mtl',
        '801720': 'Construction Dec','801730': 'Electrical Equip',
        '801740': 'Defense',        '801750': 'IT & Computer',
        '801760': 'Media',          '801770': 'Telecom',
        '801780': 'Banking',        '801790': 'Non-Bank Finance',
        '801880': 'Automotive',     '801890': 'Machinery',
    }
    prices.columns = [code_to_name.get(c, c) for c in prices.columns]

    bm = monthly_from_daily(BENCHMARK)

    # Align
    common = prices.index.intersection(bm.index)
    prices = prices.reindex(common).ffill()
    bm     = bm.reindex(common).ffill()

    monthly_rets = prices.pct_change()

    # Momentum weights (heavier short-term)
    lookbacks = [1, 3, 6]
    weights   = [0.5, 0.3, 0.2]
    max_lb    = max(lookbacks)

    # Timing: CSI300 trend filter
    bm_ma = bm.rolling(10).mean()
    timing = (bm > bm_ma).shift(1)

    results = []
    held = []

    for t in range(max_lb + 2, len(prices)):
        date = prices.index[t]
        prev = prices.index[t - 1]
        in_mkt = bool(timing.iloc[t]) if pd.notna(timing.iloc[t]) else True

        # Score each sector (previous month-end)
        scores = {}
        for col in prices.columns:
            s = 0.0
            for lb, w in zip(lookbacks, weights):
                p0 = prices.iloc[t - 1 - lb][col]
                p1 = prices.iloc[t - 1][col]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    s += w * ((p1 / p0) - 1)
            # risk-adjust
            vol = monthly_rets.iloc[max(0, t-6):t-1][col].std()
            scores[col] = s / vol if vol > 0 else s

        # Dual momentum: only consider sectors with positive absolute 1m return
        abs_ret = {col: (prices.iloc[t-1][col] / prices.iloc[t-2][col] - 1)
                   if prices.iloc[t-2][col] > 0 else -1
                   for col in prices.columns}

        if in_mkt:
            # Qualify: positive absolute momentum
            qualified = {col: sc for col, sc in scores.items()
                         if abs_ret.get(col, -1) > ABS_MOM_THRESHOLD}

            if len(qualified) >= TOP_K:
                new_held = set(sorted(qualified, key=qualified.get, reverse=True)[:TOP_K])
            elif len(qualified) > 0:
                # Fall back to defensive sectors
                new_held = set(sorted(qualified, key=qualified.get, reverse=True))
            else:
                # No positive momentum sectors → defensive rotation
                def_scores = {col: scores.get(col, -99) for col in DEFENSIVE if col in scores}
                new_held = set(sorted(def_scores, key=def_scores.get, reverse=True)[:TOP_K])
        else:
            # Full cash (market downtrend)
            if held:
                results.append({'date': date, 'ret': 0.0, 'in_market': False,
                                 'held': [], 'mode': 'cash'})
                held = []
                continue
            results.append({'date': date, 'ret': 0.0, 'in_market': False,
                             'held': [], 'mode': 'cash'})
            continue

        # Turnover cost
        sells    = set(held) - new_held
        turnover = len(sells) / TOP_K if held else 1.0
        tc_cost  = turnover * TC * 2

        # Portfolio return
        port_rets = []
        for col in new_held:
            p0_ = prices.loc[prev, col]
            p1_ = prices.loc[date, col]
            if pd.notna(p0_) and pd.notna(p1_) and p0_ > 0:
                port_rets.append((p1_ / p0_) - 1)

        gross = np.mean(port_rets) if port_rets else 0.0
        net   = gross - tc_cost
        mode  = 'offensive' if len(qualified) >= TOP_K else 'defensive'
        results.append({'date': date, 'ret': net, 'in_market': True,
                         'held': list(new_held), 'mode': mode})
        held = list(new_held)

    bt2 = pd.DataFrame(results).set_index('date')
    s2  = bt2['ret']
    bm2 = bm.pct_change().reindex(s2.index).fillna(0)

    m2   = metrics(s2,  "Dual-Mom Sector")
    mbm2 = metrics(bm2, "CSI 300")

    cov  = np.cov(s2, bm2)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
    alpha_ann = (s2.mean() - beta * bm2.mean()) * 12

    mode_counts = bt2['mode'].value_counts().to_dict() if 'mode' in bt2.columns else {}
    print(f"\n{'Metric':<22} {'Dual-Mom Sector':>16} {'CSI 300':>12}")
    print("-" * 52)
    for k, fmt in [('cagr',':.1%'),('tot',':.0%'),('vol',':.1%'),
                   ('sh',':.2f'),('mdd',':.1%'),('cal',':.2f'),('wr',':.1%')]:
        label = {'cagr':'CAGR','tot':'Total Return','vol':'Volatility',
                 'sh':'Sharpe','mdd':'Max Drawdown','cal':'Calmar','wr':'Win Rate'}[k]
        print(f"{label:<22} {format(m2[k], fmt[1:]):>16} {format(mbm2[k], fmt[1:]):>12}")
    print(f"{'Alpha/yr':<22} {alpha_ann:>+15.1%} {'—':>12}")
    print(f"{'Beta':<22} {beta:>16.2f} {'1.00':>12}")
    print(f"\nMode breakdown: {mode_counts}")

    return s2, bm2, m2, mbm2


# ─── Combined chart ────────────────────────────────────────────────────────────
def plot_all(s1, bm1, m1, mbm, bt1, s2, bm2, m2):
    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
    fig.suptitle('JoinQuant-Inspired A-Share Strategies vs 沪深300\n'
                 'Left: Multi-Index Rotation  ·  Right: Small-Cap Momentum',
                 fontsize=13, color='white', y=0.99)

    gs   = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.32,
                    left=0.07, right=0.97, top=0.93, bottom=0.05)
    dark = '#161b22'; grid = '#30363d'; txt = '#c9d1d9'
    C    = {'s1': '#58a6ff', 's2': '#3fb950', 'bm': '#f78166'}

    def sty(ax):
        ax.set_facecolor(dark)
        ax.tick_params(colors=txt, labelsize=8)
        ax.xaxis.label.set_color(txt); ax.yaxis.label.set_color(txt)
        ax.title.set_color(txt)
        for sp in ax.spines.values(): sp.set_color(grid)
        ax.grid(True, color=grid, lw=0.5, alpha=0.6)

    bm_cum = (1 + bm1).cumprod()

    # ── Row 0: Cumulative returns ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(m1['cum'].index, m1['cum'],   color=C['s1'], lw=2,
             label=f"Idx Rotation  {m1['cagr']:.1%} CAGR")
    ax1.plot(bm_cum.index, bm_cum, color=C['bm'], lw=1.5, alpha=0.8,
             label=f"CSI 300       {mbm['cagr']:.1%} CAGR")
    ax1.set_yscale('log'); ax1.set_title('Index Rotation — Cumulative Return')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.1f}x'))
    ax1.legend(facecolor=dark, edgecolor=grid, labelcolor=txt, fontsize=8)
    sty(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    bm_cum2 = (1 + bm2).cumprod()
    ax2.plot(m2['cum'].index, m2['cum'],   color=C['s2'], lw=2,
             label=f"SmallCap Mom  {m2['cagr']:.1%} CAGR")
    ax2.plot(bm_cum2.index, bm_cum2, color=C['bm'], lw=1.5, alpha=0.8,
             label=f"CSI 300       {m2['cagr']:.1%} vs {mbm['cagr']:.1%}")
    bm_ret2 = bm2
    bm_m2   = metrics(bm_ret2, 'CSI300')
    ax2.plot(bm_m2['cum'].index, bm_m2['cum'], color=C['bm'], lw=1.5, alpha=0.8)
    ax2.set_yscale('log'); ax2.set_title('Small-Cap Momentum — Cumulative Return')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.1f}x'))
    ax2.legend(facecolor=dark, edgecolor=grid, labelcolor=txt, fontsize=8)
    sty(ax2)

    # ── Row 1: Drawdown ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(m1['dd'].index, m1['dd']*100, 0, color=C['s1'], alpha=0.6)
    ax3.fill_between(mbm['dd'].index, mbm['dd']*100, 0, color=C['bm'], alpha=0.3)
    ax3.set_title(f"Drawdown  Strategy {m1['mdd']:.1%}  vs  CSI300 {mbm['mdd']:.1%}")
    sty(ax3)

    bm_dd2 = metrics(bm2, 'bm')['dd']
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(m2['dd'].index, m2['dd']*100, 0, color=C['s2'], alpha=0.6)
    ax4.fill_between(bm_dd2.index,   bm_dd2*100,   0, color=C['bm'], alpha=0.3)
    ax4.set_title(f"Drawdown  Strategy {m2['mdd']:.1%}  vs  CSI300 {m2['mdd']:.1%}")
    sty(ax4)

    # ── Row 2: Year-by-year comparison ────────────────────────────────────────
    def yearly_bar(ax, strat_rets, bm_rets, color, title):
        years = sorted(strat_rets.index.year.unique())
        ys = [(1 + strat_rets[strat_rets.index.year==y]).prod()-1 for y in years]
        yb = [(1 + bm_rets[bm_rets.index.year==y]).prod()-1 for y in years]
        x  = np.arange(len(years))
        ax.bar(x-0.2, [v*100 for v in ys], 0.4,
               color=[color if v>=0 else '#f85149' for v in ys], alpha=0.9, label='Strategy')
        ax.bar(x+0.2, [v*100 for v in yb], 0.4,
               color=['#58a6ff' if v>=0 else '#da3633' for v in yb], alpha=0.5, label='CSI 300')
        beat = sum(s>b for s,b in zip(ys,yb))
        ax.set_title(f'{title}  (Beat {beat}/{len(years)} years)')
        ax.set_xticks(x); ax.set_xticklabels(years, fontsize=7, rotation=45)
        ax.axhline(0, color=grid, lw=0.7)
        ax.legend(facecolor=dark, edgecolor=grid, labelcolor=txt, fontsize=8)
        sty(ax)

    ax5 = fig.add_subplot(gs[2, 0])
    yearly_bar(ax5, s1, bm1, C['s1'], 'Index Rotation — Year by Year')

    ax6 = fig.add_subplot(gs[2, 1])
    yearly_bar(ax6, s2, bm2, C['s2'], 'Small-Cap Momentum — Year by Year')

    out = os.path.join(OUT_DIR, 'jq_strategies.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"\nChart saved → {out}")
    plt.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Metric':<22} {'Idx Rotation':>14} {'SmCap Mom':>12} {'CSI 300':>10}")
    print("-" * 65)
    mbm_ = metrics(bm1, 'bm')
    for k, fmt in [('cagr',':.1%'),('tot',':.0%'),('vol',':.1%'),
                   ('sh',':.2f'),('mdd',':.1%'),('cal',':.2f')]:
        label = {'cagr':'CAGR','tot':'Total Return','vol':'Volatility',
                 'sh':'Sharpe','mdd':'Max Drawdown','cal':'Calmar'}[k]
        print(f"{label:<22} {format(m1[k], fmt[1:]):>14} "
              f"{format(m2[k], fmt[1:]):>12} {format(mbm_[k], fmt[1:]):>10}")
    print("=" * 65)


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    s1, bm1, m1, mbm, bt1 = strategy1()
    s2, bm2, m2, mbm2     = strategy2()
    plot_all(s1, bm1, m1, mbm, bt1, s2, bm2, m2)
