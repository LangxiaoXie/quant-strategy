"""
Detailed comparison: Enhanced Strategy vs 沪深300 (CSI 300)
- Year-by-year returns
- Alpha / Beta / Information Ratio
- Rolling metrics
- Comprehensive charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load saved monthly returns ───────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, 'monthly_returns.csv'), index_col=0, parse_dates=True)
strat = df['enhanced_strategy'].dropna()
bm    = df['csi300'].dropna()

# Align
idx   = strat.index.intersection(bm.index)
strat = strat[idx]
bm    = bm[idx]

# ─── Cumulative ───────────────────────────────────────────────────────────────
strat_cum = (1 + strat).cumprod()
bm_cum    = (1 + bm).cumprod()

# ─── Drawdowns ────────────────────────────────────────────────────────────────
def drawdown(rets):
    cum = (1 + rets).cumprod()
    return (cum - cum.cummax()) / cum.cummax()

strat_dd = drawdown(strat)
bm_dd    = drawdown(bm)

# ─── Year-by-year ─────────────────────────────────────────────────────────────
years = strat.index.year.unique()
yearly = []
for y in sorted(years):
    s = strat[strat.index.year == y]
    b = bm[bm.index.year == y]
    if len(s) == 0:
        continue
    sr = (1 + s).prod() - 1
    br = (1 + b).prod() - 1
    yearly.append({'Year': y, 'Strategy': sr, 'CSI 300': br, 'Alpha': sr - br})

yearly_df = pd.DataFrame(yearly).set_index('Year')

# ─── Risk metrics ─────────────────────────────────────────────────────────────
rf = 0.025 / 12  # 2.5% annual risk-free

# Alpha / Beta (CAPM regression)
from numpy.polynomial import polynomial as P
cov   = np.cov(strat, bm)
beta  = cov[0, 1] / cov[1, 1]
alpha_monthly = strat.mean() - beta * bm.mean()
alpha_annual  = alpha_monthly * 12

# Information Ratio
excess = strat - bm
ir     = excess.mean() / excess.std() * np.sqrt(12)

# Tracking Error
te = excess.std() * np.sqrt(12)

# Sortino Ratio (downside deviation)
downside = strat[strat < rf].std() * np.sqrt(12)
sortino  = (strat.mean() - rf) / downside * np.sqrt(12) if downside > 0 else np.inf

n = len(strat)
years_total = n / 12
cagr_strat  = strat_cum.iloc[-1] ** (1 / years_total) - 1
cagr_bm     = bm_cum.iloc[-1]    ** (1 / years_total) - 1
vol_strat   = strat.std() * np.sqrt(12)
vol_bm      = bm.std()    * np.sqrt(12)
sharpe_strat = (strat.mean() - rf) / strat.std() * np.sqrt(12)
sharpe_bm    = (bm.mean()    - rf) / bm.std()    * np.sqrt(12)
maxdd_strat = strat_dd.min()
maxdd_bm    = bm_dd.min()

# ─── Print summary ────────────────────────────────────────────────────────────
print("=" * 65)
print(f"{'Backtest Period':<30} {str(idx[0].date()):>15} → {str(idx[-1].date())}")
print(f"{'Months':<30} {n:>15}")
print("=" * 65)
print(f"{'Metric':<30} {'Enhanced':>15} {'CSI 300':>15}")
print("-" * 65)
print(f"{'CAGR':<30} {cagr_strat:>14.1%} {cagr_bm:>14.1%}")
print(f"{'Total Return':<30} {strat_cum.iloc[-1]-1:>14.0%} {bm_cum.iloc[-1]-1:>14.0%}")
print(f"{'Annualised Volatility':<30} {vol_strat:>14.1%} {vol_bm:>14.1%}")
print(f"{'Sharpe Ratio':<30} {sharpe_strat:>14.2f} {sharpe_bm:>14.2f}")
print(f"{'Sortino Ratio':<30} {sortino:>14.2f} {'—':>15}")
print(f"{'Max Drawdown':<30} {maxdd_strat:>14.1%} {maxdd_bm:>14.1%}")
print(f"{'Calmar Ratio':<30} {cagr_strat/abs(maxdd_strat):>14.2f} {cagr_bm/abs(maxdd_bm):>14.2f}")
print(f"{'Beta vs CSI 300':<30} {beta:>14.2f} {'1.00':>15}")
print(f"{'Alpha (annualised)':<30} {alpha_annual:>14.1%} {'—':>15}")
print(f"{'Information Ratio':<30} {ir:>14.2f} {'—':>15}")
print(f"{'Tracking Error':<30} {te:>14.1%} {'—':>15}")
print("=" * 65)

print("\nYear-by-Year Returns:")
print(f"{'Year':<6} {'Strategy':>10} {'CSI 300':>10} {'Alpha':>10} {'Beat?':>6}")
print("-" * 45)
for y, row in yearly_df.iterrows():
    beat = '✓' if row['Alpha'] > 0 else '✗'
    print(f"{y:<6} {row['Strategy']:>10.1%} {row['CSI 300']:>10.1%} {row['Alpha']:>10.1%} {beat:>6}")
beat_count = (yearly_df['Alpha'] > 0).sum()
print(f"\nBeat CSI 300 in {beat_count}/{len(yearly_df)} years "
      f"({beat_count/len(yearly_df):.0%})")

# ─── Plot ──────────────────────────────────────────────────────────────────────
dark_bg   = '#161b22'
grid_col  = '#30363d'
txt_col   = '#c9d1d9'
green_col = '#3fb950'
red_col   = '#f78166'
blue_col  = '#58a6ff'

def style(ax):
    ax.set_facecolor(dark_bg)
    ax.tick_params(colors=txt_col, labelsize=8)
    ax.xaxis.label.set_color(txt_col)
    ax.yaxis.label.set_color(txt_col)
    ax.title.set_color(txt_col)
    for sp in ax.spines.values():
        sp.set_color(grid_col)
    ax.grid(True, color=grid_col, lw=0.5, alpha=0.6)

fig = plt.figure(figsize=(18, 16), facecolor='#0d1117')
fig.suptitle('Enhanced Sector Rotation vs 沪深300 (CSI 300)  —  Full Backtest Analysis\n'
             f'Top-3 Sectors | Risk-Adj Momentum | Trend Filter | 0.3% Transaction Cost | '
             f'{idx[0].year}–{idx[-1].year}',
             fontsize=13, color='white', y=0.99)

gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
              left=0.07, right=0.97, top=0.94, bottom=0.04)

# ── 1. Cumulative (log) ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(strat_cum.index, strat_cum.values, color=green_col, lw=2.5,
         label=f'Enhanced Strategy  CAGR {cagr_strat:.1%}  Total {strat_cum.iloc[-1]-1:.0%}')
ax1.plot(bm_cum.index,    bm_cum.values,    color=red_col,   lw=1.8, alpha=0.85,
         label=f'CSI 300            CAGR {cagr_bm:.1%}  Total {bm_cum.iloc[-1]-1:.0%}')
ax1.set_yscale('log')
ax1.set_title('Cumulative Net Return (log scale)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}x'))
ax1.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=txt_col, fontsize=9)
# Shade cash periods
cash_mask = df.get('in_market', pd.Series(True, index=df.index)).reindex(strat.index)
cash_starts = []
in_cash = False
for i, (d, v) in enumerate(cash_mask.items()):
    if not v and not in_cash:
        in_cash = True; cash_starts.append(d)
    elif v and in_cash:
        in_cash = False
        ax1.axvspan(cash_starts[-1], d, color='#ffffff', alpha=0.05)
if in_cash:
    ax1.axvspan(cash_starts[-1], strat.index[-1], color='#ffffff', alpha=0.05)
style(ax1)

# ── 2. Drawdown ──────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(strat_dd.index, strat_dd * 100, 0, color=green_col, alpha=0.65, label='Strategy')
ax2.fill_between(bm_dd.index,    bm_dd * 100,    0, color=red_col,   alpha=0.40, label='CSI 300')
ax2.set_title(f'Drawdown  (Strategy max {maxdd_strat:.1%} vs CSI 300 {maxdd_bm:.1%})')
ax2.set_ylabel('%')
ax2.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=txt_col, fontsize=9)
style(ax2)

# ── 3. Rolling 12m Sharpe ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
roll_sharpe_s = strat.rolling(12).apply(lambda r: (r.mean() - rf) / r.std() * np.sqrt(12) if r.std() > 0 else 0)
roll_sharpe_b = bm.rolling(12).apply(  lambda r: (r.mean() - rf) / r.std() * np.sqrt(12) if r.std() > 0 else 0)
ax3.plot(roll_sharpe_s.index, roll_sharpe_s, color=green_col, lw=1.5, label='Strategy')
ax3.plot(roll_sharpe_b.index, roll_sharpe_b, color=red_col,   lw=1.2, alpha=0.8, label='CSI 300')
ax3.axhline(0, color=grid_col, lw=0.8)
ax3.axhline(1, color='#ffffff', lw=0.5, linestyle='--', alpha=0.3)
ax3.set_title('Rolling 12-Month Sharpe Ratio')
ax3.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=txt_col, fontsize=9)
style(ax3)

# ── 4. Year-by-year bar chart ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :])
x   = np.arange(len(yearly_df))
w   = 0.35
bars_s = ax4.bar(x - w/2, yearly_df['Strategy'] * 100, w, label='Strategy',
                 color=[green_col if v >= 0 else '#f85149' for v in yearly_df['Strategy']], alpha=0.85)
bars_b = ax4.bar(x + w/2, yearly_df['CSI 300'] * 100,  w, label='CSI 300',
                 color=[blue_col  if v >= 0 else '#da3633' for v in yearly_df['CSI 300']],  alpha=0.65)
ax4.axhline(0, color=txt_col, lw=0.6)
ax4.set_xticks(x)
ax4.set_xticklabels(yearly_df.index, fontsize=8, rotation=45)
ax4.set_ylabel('Annual Return (%)')
ax4.set_title('Year-by-Year Returns: Strategy vs 沪深300')
ax4.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=txt_col, fontsize=9)
# Annotate alpha
for i, (y, row) in enumerate(yearly_df.iterrows()):
    a = row['Alpha'] * 100
    col = green_col if a > 0 else red_col
    ax4.text(i, max(row['Strategy'], row['CSI 300']) * 100 + 1.5,
             f'+{a:.0f}%' if a > 0 else f'{a:.0f}%',
             ha='center', va='bottom', fontsize=6.5, color=col)
style(ax4)

# ── 5. Excess return (strategy − CSI300) cumulative ─────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
excess_cum = (1 + (strat - bm)).cumprod() - 1
ax5.plot(excess_cum.index, excess_cum * 100, color='#d2a8ff', lw=1.8)
ax5.fill_between(excess_cum.index, excess_cum * 100, 0,
                 where=excess_cum >= 0, color='#3fb950', alpha=0.25)
ax5.fill_between(excess_cum.index, excess_cum * 100, 0,
                 where=excess_cum <  0, color=red_col,   alpha=0.25)
ax5.axhline(0, color=grid_col, lw=0.8)
ax5.set_title(f'Cumulative Excess Return vs 沪深300  (IR={ir:.2f})')
ax5.set_ylabel('%')
style(ax5)

# ── 6. Monthly return scatter ────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
colors = [green_col if s > b else red_col for s, b in zip(strat, bm)]
ax6.scatter(bm * 100, strat * 100, c=colors, alpha=0.55, s=18, edgecolors='none')
lims = [-25, 30]
ax6.plot(lims, lims, '--', color=grid_col, lw=1, label='y=x')
# Regression line
m, c_ = np.polyfit(bm, strat, 1)
xs = np.linspace(bm.min(), bm.max(), 100)
ax6.plot(xs * 100, (m * xs + c_) * 100, color='#d2a8ff', lw=1.5,
         label=f'β={m:.2f}, α={c_*12:.1%}/yr')
ax6.set_xlabel('CSI 300 Monthly Return (%)')
ax6.set_ylabel('Strategy Monthly Return (%)')
ax6.set_title('Monthly Return Scatter (green = strategy beat)')
ax6.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=txt_col, fontsize=8)
style(ax6)

out = os.path.join(OUT_DIR, 'comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"\nChart saved → {out}")
plt.close()
