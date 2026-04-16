"""
A-Share Multi-Strategy Ensemble
================================
Based on insights from Chinese retail quant community (知乎):
combining uncorrelated strategies reduces drawdown and smooths returns.

New Strategies
--------------
Strategy 3: 低波动轮动 (Low-Volatility Sector Rotation)
  - Select top-K sectors with LOWEST trailing 12m volatility
  - Inverse-vol weighting (lower vol → higher weight)
  - Timing: CSI300 > 10m MA

Strategy 4: 均值回归 (Mean-Reversion / Contrarian)
  - Select sectors with WORST 1m return but BEST 6m return
  - Bets on short-term pullbacks within ongoing uptrends
  - Timing: same trend filter

Strategy 5: 多策略组合 (Equal-Weight Ensemble)
  - Equal-weight blend of all 5 strategy return series
  - Theoretically optimal for diversifying strategy-specific risk
  - Target: highest Sharpe, lowest drawdown
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import akshare as ak

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RF = 0.025 / 12
TC = 0.003

CODE_TO_NAME = {
    '801010':'Agriculture','801020':'Mining','801030':'Chemicals',
    '801040':'Steel','801050':'Non-Ferrous','801080':'Electronics',
    '801110':'Home Appliances','801120':'Food & Beverage','801130':'Textiles',
    '801140':'Light Mfg','801150':'Pharma','801160':'Utilities',
    '801170':'Transport','801180':'Real Estate','801200':'Commerce',
    '801210':'Leisure','801230':'Conglomerates','801710':'Construction Mtl',
    '801720':'Construction Dec','801730':'Electrical Equip','801740':'Defense',
    '801750':'IT & Computer','801760':'Media','801770':'Telecom',
    '801780':'Banking','801790':'Non-Bank Finance','801880':'Automotive',
    '801890':'Machinery',
}
DEFENSIVE = ['Banking', 'Utilities', 'Food & Beverage']

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_sectors() -> pd.DataFrame:
    cache = os.path.join(OUT_DIR, 'prices_cache.csv')
    df = pd.read_csv(cache, index_col=0, parse_dates=True).ffill()
    df.columns = [CODE_TO_NAME.get(c, c) for c in df.columns]
    return df

def load_bm() -> pd.Series:
    df = ak.stock_zh_index_daily(symbol='sh000300')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    s = df['close'].resample('ME').last()
    s.index = s.index.to_period('M').to_timestamp('M')
    return s

# ─── Metrics ──────────────────────────────────────────────────────────────────
def perf(rets, label=''):
    n = len(rets); c = (1+rets).cumprod()
    tot = c.iloc[-1]-1; cagr = (1+tot)**(12/n)-1
    vol = rets.std()*np.sqrt(12)
    sh  = (rets.mean()-RF)/rets.std()*np.sqrt(12) if rets.std()>0 else 0
    dd  = (c - c.cummax())/c.cummax(); mdd = dd.min()
    cal = cagr/abs(mdd) if mdd else 0
    cov = np.cov(rets, rets); beta = 1.0  # placeholder
    return dict(label=label,cagr=cagr,tot=tot,vol=vol,sh=sh,mdd=mdd,
                cal=cal,n=n,cum=c,dd=dd)

def alpha_beta(s, b):
    idx = s.index.intersection(b.index)
    sv, bv = s[idx].values, b[idx].values
    cov = np.cov(sv, bv)
    beta = cov[0,1]/cov[1,1] if cov[1,1]>0 else 0
    alpha = (sv.mean() - beta*bv.mean())*12
    return alpha, beta

def print_table(strategies: dict, bm_rets: pd.Series):
    """Print comparison table for all strategies."""
    bm_m = perf(bm_rets, 'CSI 300')
    cols = list(strategies.keys()) + ['CSI 300']
    metrics_map = {}
    for name, rets in strategies.items():
        b = bm_rets.reindex(rets.index).fillna(0)
        m = perf(rets, name)
        a, beta = alpha_beta(rets, b)
        m['alpha'] = a; m['beta'] = beta
        metrics_map[name] = m
    a_bm, _ = alpha_beta(bm_rets, bm_rets)
    bm_m['alpha'] = 0; bm_m['beta'] = 1.0
    metrics_map['CSI 300'] = bm_m

    print("\n" + "="*95)
    header = f"{'Metric':<20}" + "".join(f"{c:>15}" for c in cols)
    print(header)
    print("-"*95)
    for key, fmt, label in [
        ('cagr', ':.1%', 'CAGR'),
        ('tot',  ':.0%', 'Total Return'),
        ('vol',  ':.1%', 'Volatility'),
        ('sh',   ':.2f', 'Sharpe'),
        ('mdd',  ':.1%', 'Max Drawdown'),
        ('cal',  ':.2f', 'Calmar'),
        ('alpha',':.1%', 'Alpha/yr'),
        ('beta', ':.2f', 'Beta'),
    ]:
        row = f"{label:<20}"
        for c in cols:
            v = metrics_map[c][key]
            row += f"{format(v, fmt[1:]):>15}"
        print(row)
    print("="*95)

# ─── Strategy engines ─────────────────────────────────────────────────────────

def run_original_sector(prices, bm_rets, top_n=3, trend_win=10, start_year=2005):
    """Strategy 1: Risk-adjusted composite momentum (original)."""
    prices = prices[prices.index.year >= start_year].copy()
    bm     = bm_rets[bm_rets.index.year >= start_year].copy()
    mr = prices.pct_change()
    bm_lvl = (1+bm).cumprod()
    bm_aln = bm_lvl.reindex(prices.index, method='ffill')
    trend  = (bm_aln > bm_aln.rolling(trend_win).mean()).shift(1)
    lookbacks=[1,3,6,12]; weights=[.4,.3,.2,.1]
    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for t in range(max(lookbacks)+1, len(prices)):
        row={}
        for col in prices.columns:
            s=0.
            for lb,w in zip(lookbacks,weights):
                p0,p1=prices.iloc[t-lb][col],prices.iloc[t][col]
                if pd.notna(p0) and pd.notna(p1) and p0>0: s+=w*((p1/p0)-1)
            vol=mr.iloc[max(0,t-12):t][col].std()
            row[col]=s/vol if vol>0 else s
        scores.iloc[t]=pd.Series(row)
    results=[]; held=set()
    for t in range(max(lookbacks)+2, len(prices)):
        date,prev=prices.index[t],prices.index[t-1]
        in_mkt=bool(trend.iloc[t]) if pd.notna(trend.iloc[t]) else True
        if not in_mkt:
            results.append({'date':date,'ret':0.}); held=set(); continue
        row_s=scores.iloc[t-1].dropna()
        if len(row_s)<top_n:
            results.append({'date':date,'ret':0.}); continue
        new_held=set(row_s.nlargest(top_n).index)
        turn=len(held-new_held)/top_n if held else 1.
        pr=[(prices.loc[date,c]/prices.loc[prev,c])-1 for c in new_held
            if pd.notna(prices.loc[prev,c]) and prices.loc[prev,c]>0]
        results.append({'date':date,'ret':(np.mean(pr) if pr else 0.)-turn*TC*2}); held=new_held
    bt=pd.DataFrame(results).set_index('date')
    return bt['ret'], bm.reindex(bt.index).fillna(0)


def run_low_vol(prices, bm_rets, top_k=3, vol_window=12, trend_win=10, start_year=2005):
    """
    Strategy 3: 低波动轮动 (Low-Volatility Sector Rotation)

    Selects the K sectors with lowest trailing volatility.
    Weights positions inversely proportional to volatility (min-var tilt).
    Logic: low-vol sectors outperform on a risk-adjusted basis in A-shares
    because retail-driven market has vol premium in small/speculative sectors.
    """
    prices = prices[prices.index.year >= start_year].copy()
    bm     = bm_rets[bm_rets.index.year >= start_year].copy()
    mr = prices.pct_change()
    bm_lvl = (1+bm).cumprod()
    bm_aln = bm_lvl.reindex(prices.index, method='ffill')
    trend  = (bm_aln > bm_aln.rolling(trend_win).mean()).shift(1)

    results=[]; held=set()
    for t in range(vol_window+2, len(prices)):
        date,prev=prices.index[t],prices.index[t-1]
        in_mkt=bool(trend.iloc[t]) if pd.notna(trend.iloc[t]) else True
        if not in_mkt:
            results.append({'date':date,'ret':0.,'mode':'cash'}); held=set(); continue

        # Trailing volatility for each sector
        vols = {}
        for col in prices.columns:
            v = mr.iloc[max(0,t-vol_window):t][col].std()
            if pd.notna(v) and v>0: vols[col] = v

        if len(vols) < top_k:
            results.append({'date':date,'ret':0.,'mode':'insufficient'}); continue

        # Select lowest-vol sectors
        sorted_by_vol = sorted(vols, key=vols.get)
        new_held = set(sorted_by_vol[:top_k])

        # Inverse-vol weights (min-variance tilt)
        inv_vols = {c: 1.0/vols[c] for c in new_held}
        total_inv = sum(inv_vols.values())
        weights  = {c: inv_vols[c]/total_inv for c in new_held}

        turn = len(held - new_held)/top_k if held else 1.
        tc_cost = turn * TC * 2

        pr = 0.
        for col, w in weights.items():
            p0 = prices.loc[prev, col]; p1 = prices.loc[date, col]
            if pd.notna(p0) and pd.notna(p1) and p0>0:
                pr += w * ((p1/p0)-1)

        results.append({'date':date,'ret':pr-tc_cost,'mode':'invested'}); held=new_held

    bt=pd.DataFrame(results).set_index('date')
    return bt['ret'], bm.reindex(bt.index).fillna(0)


def run_mean_reversion(prices, bm_rets, top_k=3, trend_win=10, start_year=2005):
    """
    Strategy 4: 均值回归轮动 (Mean-Reversion / Contrarian Rotation)

    Selects sectors that have pulled back SHORT-term (worst 1m return)
    but remain in medium-term uptrend (positive 6m return).

    Logic: In trending markets, strong sectors pull back temporarily.
    Buying the dip in confirmed uptrend sectors captures mean-reversion premium.
    This is uncorrelated with pure momentum → good ensemble component.
    """
    prices = prices[prices.index.year >= start_year].copy()
    bm     = bm_rets[bm_rets.index.year >= start_year].copy()
    bm_lvl = (1+bm).cumprod()
    bm_aln = bm_lvl.reindex(prices.index, method='ffill')
    trend  = (bm_aln > bm_aln.rolling(trend_win).mean()).shift(1)

    results=[]; held=set()
    for t in range(8, len(prices)):
        date,prev=prices.index[t],prices.index[t-1]
        in_mkt=bool(trend.iloc[t]) if pd.notna(trend.iloc[t]) else True
        if not in_mkt:
            results.append({'date':date,'ret':0.}); held=set(); continue

        # 1m return (short-term, we want the LOWEST = most oversold)
        ret_1m = {}
        # 6m return (medium-term, must be POSITIVE = in uptrend)
        ret_6m = {}
        for col in prices.columns:
            if t>=1:
                p0,p1=prices.iloc[t-2][col],prices.iloc[t-1][col]
                if pd.notna(p0) and pd.notna(p1) and p0>0: ret_1m[col]=(p1/p0)-1
            if t>=6:
                p0,p1=prices.iloc[t-7][col],prices.iloc[t-1][col]
                if pd.notna(p0) and pd.notna(p1) and p0>0: ret_6m[col]=(p1/p0)-1

        # Filter: only sectors with positive 6m return (confirmed uptrend)
        qualified = {c: ret_1m[c] for c in ret_1m
                     if c in ret_6m and ret_6m[c] > 0.03}  # >3% 6m trend

        if len(qualified) < top_k:
            # Fall back to defensive sectors
            def_rets = {c: ret_1m.get(c, 0) for c in DEFENSIVE if c in ret_1m}
            if def_rets:
                qualified = def_rets
            else:
                results.append({'date':date,'ret':0.}); continue

        # Buy the most oversold (lowest 1m return) among qualified
        new_held = set(sorted(qualified, key=qualified.get)[:top_k])

        turn = len(held-new_held)/top_k if held else 1.
        pr=[(prices.loc[date,c]/prices.loc[prev,c])-1 for c in new_held
            if pd.notna(prices.loc[prev,c]) and prices.loc[prev,c]>0]
        results.append({'date':date,'ret':(np.mean(pr) if pr else 0.)-turn*TC*2})
        held=new_held

    bt=pd.DataFrame(results).set_index('date')
    return bt['ret'], bm.reindex(bt.index).fillna(0)


def run_ensemble(all_rets: dict, bm_rets: pd.Series):
    """
    Strategy 5: 多策略等权组合 (Equal-Weight Ensemble)

    Blends all strategy return series with equal weight.
    Theory: if strategies have low pairwise correlation,
    the ensemble Sharpe ≈ average_Sharpe × √n_strategies.
    Minimises single-strategy blow-up risk.
    """
    # Align all series to common dates
    common = None
    for name, s in all_rets.items():
        common = s.index if common is None else common.intersection(s.index)

    aligned = {name: s.reindex(common).fillna(0) for name, s in all_rets.items()}
    ensemble = pd.concat(aligned.values(), axis=1).mean(axis=1)
    ensemble.name = 'Ensemble'
    bm_aligned = bm_rets.reindex(common).fillna(0)
    return ensemble, bm_aligned


# ─── Correlation analysis ─────────────────────────────────────────────────────
def correlation_matrix(strategies: dict) -> pd.DataFrame:
    """Return monthly return correlation matrix across strategies."""
    common = None
    for s in strategies.values():
        common = s.index if common is None else common.intersection(s.index)
    df = pd.concat({k: v.reindex(common).fillna(0) for k,v in strategies.items()}, axis=1)
    return df.corr()


# ─── Plotting ─────────────────────────────────────────────────────────────────
PALETTE = {
    'Momentum':     '#58a6ff',
    'Low-Vol':      '#3fb950',
    'Mean-Rev':     '#e3b341',
    'Dual-Mom':     '#a371f7',
    'Ensemble':     '#ff6b6b',
    'CSI 300':      '#666',
}

def plot_ensemble(strategies: dict, bm_rets: pd.Series):
    """Six-panel comparison chart."""
    # Align everything
    common = None
    for s in strategies.values():
        common = s.index if common is None else common.intersection(s.index)
    bm = bm_rets.reindex(common).fillna(0)

    fig = plt.figure(figsize=(20, 16), facecolor='#0d1117')
    fig.suptitle('A-Share Multi-Strategy Ensemble\n'
                 '5 Strategies + Equal-Weight Blend  ·  28 Shenwan Sectors  ·  Monthly Rebalance',
                 fontsize=14, color='white', y=0.99)

    gs = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32,
                  left=0.07, right=0.97, top=0.93, bottom=0.05)
    dark='#161b22'; grid='#30363d'; txt='#c9d1d9'

    def sty(ax):
        ax.set_facecolor(dark); ax.tick_params(colors=txt, labelsize=8)
        ax.xaxis.label.set_color(txt); ax.yaxis.label.set_color(txt)
        ax.title.set_color(txt)
        for sp in ax.spines.values(): sp.set_color(grid)
        ax.grid(True, color=grid, lw=0.5, alpha=0.6)

    # ── 1. Cumulative returns (all) ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    bm_cum = (1+bm).cumprod()
    ax1.plot(bm_cum.index, bm_cum, color=PALETTE['CSI 300'], lw=1.5,
             alpha=0.7, label=f"CSI 300  {perf(bm)['cagr']:.1%} CAGR", linestyle='--')
    for name, rets in strategies.items():
        r = rets.reindex(common).fillna(0)
        cum = (1+r).cumprod()
        lw = 3 if name=='Ensemble' else 1.8
        ax1.plot(cum.index, cum, color=PALETTE.get(name,'#aaa'), lw=lw,
                 label=f"{name}  {perf(r)['cagr']:.1%}")
    ax1.set_yscale('log'); ax1.set_title('Cumulative Net Return (log scale)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.1f}x'))
    ax1.legend(facecolor=dark, edgecolor=grid, labelcolor=txt, fontsize=8,
               ncol=3, loc='upper left')
    sty(ax1)

    # ── 2. Drawdown comparison ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bm_dd = (bm_cum - bm_cum.cummax()) / bm_cum.cummax()
    ax2.fill_between(bm_dd.index, bm_dd*100, 0, color=PALETTE['CSI 300'], alpha=0.25)
    for name, rets in strategies.items():
        r = rets.reindex(common).fillna(0)
        c = (1+r).cumprod(); dd = (c - c.cummax())/c.cummax()
        lw = 2.5 if name=='Ensemble' else 1.2
        ax2.plot(dd.index, dd*100, color=PALETTE.get(name,'#aaa'), lw=lw,
                 alpha=0.9 if name=='Ensemble' else 0.65, label=name)
    ax2.set_title('Drawdown Comparison'); ax2.set_ylabel('%')
    ax2.legend(facecolor=dark, edgecolor=grid, labelcolor=txt, fontsize=7)
    sty(ax2)

    # ── 3. Correlation heatmap ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    corr_df = correlation_matrix(strategies)
    n = len(corr_df)
    im = ax3.imshow(corr_df.values, cmap='RdYlGn', vmin=-1, vmax=1,
                    aspect='auto', interpolation='nearest')
    ax3.set_xticks(range(n)); ax3.set_yticks(range(n))
    ax3.set_xticklabels(corr_df.columns, fontsize=8, rotation=30, ha='right',
                        color=txt)
    ax3.set_yticklabels(corr_df.index, fontsize=8, color=txt)
    for i in range(n):
        for j in range(n):
            v = corr_df.values[i,j]
            ax3.text(j, i, f'{v:.2f}', ha='center', va='center',
                     fontsize=9, color='black' if abs(v)<0.6 else 'white',
                     fontweight='bold')
    ax3.set_title('Strategy Return Correlations')
    ax3.set_facecolor(dark)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    for sp in ax3.spines.values(): sp.set_color(grid)

    # ── 4. Year-by-year: ensemble vs CSI300 ───────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ens = strategies['Ensemble'].reindex(common).fillna(0)
    years = sorted(ens.index.year.unique())
    ys = [(1+ens[ens.index.year==y]).prod()-1 for y in years]
    yb = [(1+bm[bm.index.year==y]).prod()-1 for y in years]
    x  = np.arange(len(years))
    ax4.bar(x-.2, [v*100 for v in ys], .38,
            color=['#ff6b6b' if v>=0 else '#f85149' for v in ys], alpha=0.9, label='Ensemble')
    ax4.bar(x+.2, [v*100 for v in yb], .38,
            color=['rgba(88,166,255,0.5)' if False else '#3a5a7a' if v>=0 else '#6b2525' for v in yb],
            alpha=0.7, label='CSI 300')
    beat = sum(s>b_ for s,b_ in zip(ys,yb))
    ax4.set_title(f'Ensemble Year-by-Year  (Beat {beat}/{len(years)} = {beat/len(years):.0%})')
    ax4.set_xticks(x); ax4.set_xticklabels(years, fontsize=7, rotation=45)
    ax4.axhline(0, color=grid, lw=.7)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:.0f}%'))
    ax4.legend(facecolor=dark, edgecolor=grid, labelcolor=txt, fontsize=8)
    sty(ax4)

    # ── 5. Rolling 12m Sharpe: ensemble vs strategies ─────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    for name, rets in strategies.items():
        r = rets.reindex(common).fillna(0)
        roll = r.rolling(12).apply(
            lambda x: (x.mean()-RF)/x.std()*np.sqrt(12) if x.std()>0 else 0)
        lw = 2.5 if name=='Ensemble' else 1.0
        ax5.plot(roll.index, roll, color=PALETTE.get(name,'#aaa'), lw=lw,
                 alpha=1 if name=='Ensemble' else 0.55, label=name)
    bm_roll = bm.rolling(12).apply(
        lambda x: (x.mean()-RF)/x.std()*np.sqrt(12) if x.std()>0 else 0)
    ax5.plot(bm_roll.index, bm_roll, color=PALETTE['CSI 300'], lw=1,
             alpha=0.6, linestyle='--', label='CSI 300')
    ax5.axhline(0, color=grid, lw=.7)
    ax5.axhline(1, color='#fff', lw=.5, linestyle=':', alpha=0.3)
    ax5.set_title('Rolling 12-Month Sharpe Ratio')
    ax5.legend(facecolor=dark, edgecolor=grid, labelcolor=txt, fontsize=7,
               ncol=2)
    sty(ax5)

    out = os.path.join(OUT_DIR, 'ensemble_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Chart → {out}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    prices = load_sectors()
    bm_daily = load_bm()
    bm_rets  = bm_daily.pct_change().dropna()

    print("\n── Strategy 1: Risk-Adj Momentum (original) ──")
    s1, b1 = run_original_sector(prices, bm_rets, top_n=3, start_year=2005)

    print("── Strategy 3: Low-Volatility Rotation ──")
    s3, b3 = run_low_vol(prices, bm_rets, top_k=3, start_year=2005)

    print("── Strategy 4: Mean-Reversion Rotation ──")
    s4, b4 = run_mean_reversion(prices, bm_rets, top_k=3, start_year=2005)

    # Load existing dual-momentum results from saved CSV
    print("── Loading Strategy 2 (Dual-Momentum) from CSV ──")
    mr_csv = os.path.join(OUT_DIR, 'monthly_returns.csv')
    mr_df  = pd.read_csv(mr_csv, index_col=0, parse_dates=True)
    s2 = mr_df['enhanced_strategy'].dropna()
    s2.name = 'Dual-Mom'

    print("── Strategy 5: Equal-Weight Ensemble ──")
    # Blend all 4 strategies
    all_strategies = {
        'Momentum': s1,
        'Dual-Mom': s2,
        'Low-Vol':  s3,
        'Mean-Rev': s4,
    }
    s5, b5 = run_ensemble(all_strategies, bm_rets)

    strategies_full = {**all_strategies, 'Ensemble': s5}

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr = correlation_matrix(strategies_full)
    print("\n── Pairwise Correlations ──")
    print(corr.round(3).to_string())

    # ── Performance table ─────────────────────────────────────────────────────
    print_table(strategies_full, bm_rets)

    # ── Chart ─────────────────────────────────────────────────────────────────
    print("\nGenerating chart...")
    plot_ensemble(strategies_full, bm_rets)

    # ── Save ensemble CSV ─────────────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, 'ensemble_returns.csv')
    common = s5.index
    pd.DataFrame({
        'momentum':  s1.reindex(common).fillna(0),
        'dual_mom':  s2.reindex(common).fillna(0),
        'low_vol':   s3.reindex(common).fillna(0),
        'mean_rev':  s4.reindex(common).fillna(0),
        'ensemble':  s5,
        'csi300':    bm_rets.reindex(common).fillna(0),
    }).to_csv(out_csv)
    print(f"Returns CSV → {out_csv}")
