"""
A-Share Sector Rotation Dashboard
Streamlit app — live data via akshare
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import akshare as ak

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A-Share Sector Rotation",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 16px 20px; text-align: center;
    }
    .metric-label { font-size: 12px; color: #8b949e; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 700; }
    .positive { color: #3fb950; }
    .negative { color: #f85149; }
    .neutral  { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
SW_INDUSTRIES = {
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

# ─── Data loading (cached 24h) ────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def load_sector_prices() -> pd.DataFrame:
    series = []
    for code, name in SW_INDUSTRIES.items():
        try:
            df = ak.index_hist_sw(symbol=code, period='month')
            s = df[['日期', '收盘']].copy()
            s['日期'] = pd.to_datetime(s['日期'])
            s = s.set_index('日期').sort_index()['收盘'].rename(name)
            series.append(s)
        except Exception:
            pass
        time.sleep(0.2)
    prices = pd.concat(series, axis=1)
    prices.index = pd.to_datetime(prices.index)
    return prices.ffill()


@st.cache_data(ttl=86400, show_spinner=False)
def load_benchmark() -> pd.Series:
    df = ak.stock_zh_index_daily(symbol='sh000300')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    monthly = df['close'].resample('ME').last()
    monthly.index = monthly.index.to_period('M').to_timestamp('M')
    return monthly.rename('CSI300')


# ─── Strategy logic ────────────────────────────────────────────────────────────
def run_strategy(prices: pd.DataFrame, benchmark: pd.Series,
                 top_n: int, lookbacks: list, weights: list,
                 trend_window: int, tc: float,
                 risk_adj: bool, start_year: int) -> dict:

    prices = prices[prices.index.year >= start_year].copy()
    bm     = benchmark[benchmark.index.year >= start_year].copy()

    monthly_rets = prices.pct_change()

    # Trend filter
    bm_aligned = bm.reindex(prices.index, method='ffill')
    ma = bm_aligned.rolling(trend_window).mean()
    trend = (bm_aligned > ma).shift(1)

    # Score matrix
    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    max_lb = max(lookbacks)
    for t in range(max_lb + 1, len(prices)):
        row = {}
        for col in prices.columns:
            s = 0.0
            for lb, w in zip(lookbacks, weights):
                r = (prices.iloc[t][col] / prices.iloc[t - lb][col]) - 1
                if not np.isnan(r):
                    s += w * r
            if risk_adj:
                vol = monthly_rets.iloc[max(0, t-12):t][col].std()
                if vol > 0 and not np.isnan(vol):
                    s /= vol
            row[col] = s
        scores.iloc[t] = pd.Series(row)

    # Backtest
    results = []
    held = set()
    for t in range(max_lb + 2, len(prices)):
        date = prices.index[t]
        prev = prices.index[t - 1]
        in_mkt = bool(trend.iloc[t]) if pd.notna(trend.iloc[t]) else True

        if not in_mkt:
            turnover = 1.0 if held else 0.0
            held = set()
            results.append({'date': date, 'ret': 0.0, 'turnover': turnover, 'in_market': False, 'held': []})
            continue

        row_s = scores.iloc[t - 1].dropna()
        if len(row_s) < top_n:
            results.append({'date': date, 'ret': 0.0, 'turnover': 0.0, 'in_market': True, 'held': list(held)})
            continue

        new_held = set(row_s.nlargest(top_n).index)
        turnover = len(held - new_held) / top_n if held else 1.0

        port_rets = []
        for col in new_held:
            p0, p1 = prices.loc[prev, col], prices.loc[date, col]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                port_rets.append((p1 / p0) - 1)

        gross = np.mean(port_rets) if port_rets else 0.0
        net   = gross - turnover * tc * 2
        results.append({'date': date, 'ret': net, 'turnover': turnover,
                        'in_market': True, 'held': list(new_held)})
        held = new_held

    bt = pd.DataFrame(results).set_index('date')
    strat_rets = bt['ret']
    bm_rets    = bm.pct_change().reindex(strat_rets.index).fillna(0)

    return {
        'strat_rets': strat_rets,
        'bm_rets':    bm_rets,
        'bt':         bt,
        'prices':     prices,
        'scores':     scores,
    }


def metrics(rets: pd.Series, label: str) -> dict:
    rf = 0.025 / 12
    n  = len(rets)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr  = (1 + total) ** (12 / n) - 1
    vol   = rets.std() * np.sqrt(12)
    sharpe = (rets.mean() - rf) / rets.std() * np.sqrt(12) if rets.std() > 0 else 0
    dd = (cum - cum.cummax()) / cum.cummax()
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.inf
    return dict(label=label, cagr=cagr, total=total, vol=vol,
                sharpe=sharpe, mdd=mdd, calmar=calmar, n=n, cum=cum, dd=dd)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")
    top_n       = st.slider("Top-N sectors to hold", 1, 8, 3)
    trend_win   = st.slider("Trend filter window (months)", 3, 24, 10)
    tc_pct      = st.slider("Transaction cost (one-way %)", 0.0, 1.0, 0.3, 0.05)
    start_year  = st.slider("Start year", 2005, 2015, 2006)
    risk_adj    = st.checkbox("Risk-adjusted momentum", value=True)
    st.markdown("---")
    st.caption("Lookback weights: 1m×0.4 + 3m×0.3 + 6m×0.2 + 12m×0.1")
    st.caption("Trend filter: invest only when CSI 300 > N-month MA")
    st.caption("Data: Shenwan Level-1 indices via akshare")

# ─── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading sector data... (cached after first run)"):
    prices    = load_sector_prices()
    benchmark = load_benchmark()

# ─── Run ──────────────────────────────────────────────────────────────────────
res = run_strategy(prices, benchmark,
                   top_n=top_n,
                   lookbacks=[1, 3, 6, 12],
                   weights=[0.4, 0.3, 0.2, 0.1],
                   trend_window=trend_win,
                   tc=tc_pct / 100,
                   risk_adj=risk_adj,
                   start_year=start_year)

sm = metrics(res['strat_rets'], 'Strategy')
bm = metrics(res['bm_rets'],    'CSI 300')

# Alpha / Beta
cov  = np.cov(res['strat_rets'], res['bm_rets'])
beta = cov[0, 1] / cov[1, 1]
alpha_ann = (res['strat_rets'].mean() - beta * res['bm_rets'].mean()) * 12

excess = res['strat_rets'] - res['bm_rets']
ir     = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📈 A-Share Sector Rotation Strategy")
st.caption(f"Backtest: {res['strat_rets'].index[0].strftime('%Y-%m')} → "
           f"{res['strat_rets'].index[-1].strftime('%Y-%m')}  ·  "
           f"{sm['n']} months  ·  Top-{top_n} Shenwan L1 sectors  ·  "
           f"Monthly rebalance  ·  {tc_pct:.2f}% one-way cost")

# ─── KPI row ──────────────────────────────────────────────────────────────────
cols = st.columns(7)
kpis = [
    ("CAGR", f"{sm['cagr']:.1%}", f"vs {bm['cagr']:.1%}", sm['cagr'] > bm['cagr']),
    ("Total Return", f"{sm['total']:.0%}", f"vs {bm['total']:.0%}", sm['total'] > bm['total']),
    ("Sharpe", f"{sm['sharpe']:.2f}", f"vs {bm['sharpe']:.2f}", sm['sharpe'] > bm['sharpe']),
    ("Max Drawdown", f"{sm['mdd']:.1%}", f"vs {bm['mdd']:.1%}", sm['mdd'] > bm['mdd']),
    ("Alpha/yr", f"+{alpha_ann:.1%}" if alpha_ann > 0 else f"{alpha_ann:.1%}", "vs benchmark", alpha_ann > 0),
    ("Beta", f"{beta:.2f}", "market exposure", None),
    ("Info Ratio", f"{ir:.2f}", "excess/tracking err", ir > 0),
]
for col, (label, val, sub, pos) in zip(cols, kpis):
    color_cls = "positive" if pos is True else ("negative" if pos is False else "neutral")
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_cls}">{val}</div>
        <div class="metric-label">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Chart 1: Cumulative returns ──────────────────────────────────────────────
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=sm['cum'].index, y=sm['cum'].values,
    name=f"Strategy (CAGR {sm['cagr']:.1%})", line=dict(color='#3fb950', width=2.5)))
fig1.add_trace(go.Scatter(x=bm['cum'].index, y=bm['cum'].values,
    name=f"CSI 300 (CAGR {bm['cagr']:.1%})",  line=dict(color='#f78166', width=1.8)))

# Shade cash periods
cash_dates = res['bt'][~res['bt']['in_market']].index
if len(cash_dates):
    in_cash = False; start_cash = None
    for d, row in res['bt'].iterrows():
        if not row['in_market'] and not in_cash:
            in_cash = True; start_cash = d
        elif row['in_market'] and in_cash:
            in_cash = False
            fig1.add_vrect(x0=start_cash, x1=d, fillcolor='white', opacity=0.04, line_width=0)
    if in_cash:
        fig1.add_vrect(x0=start_cash, x1=res['bt'].index[-1], fillcolor='white', opacity=0.04, line_width=0)

fig1.update_layout(
    title="Cumulative Net Return (log scale) — shaded = cash (trend filter active)",
    yaxis_type="log", yaxis_tickformat=".1f",
    yaxis_title="Growth of ¥1 (log)",
    template="plotly_dark", height=380,
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=50, r=20, t=50, b=40),
)
st.plotly_chart(fig1, use_container_width=True)

# ─── Chart 2+3: Drawdown & Rolling Sharpe ─────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sm['dd'].index, y=sm['dd'].values * 100,
        fill='tozeroy', name='Strategy', line=dict(color='#3fb950', width=1.5),
        fillcolor='rgba(63,185,80,0.25)'))
    fig2.add_trace(go.Scatter(x=bm['dd'].index, y=bm['dd'].values * 100,
        fill='tozeroy', name='CSI 300',  line=dict(color='#f78166', width=1.2),
        fillcolor='rgba(248,113,102,0.15)'))
    fig2.update_layout(
        title=f"Drawdown  (Strategy {sm['mdd']:.1%} vs CSI300 {bm['mdd']:.1%})",
        yaxis_ticksuffix='%', template='plotly_dark', height=300,
        margin=dict(l=50, r=20, t=50, b=40), legend=dict(x=0.01, y=0.05))
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    rf_m = 0.025 / 12
    roll_s = res['strat_rets'].rolling(12).apply(
        lambda r: (r.mean()-rf_m)/r.std()*np.sqrt(12) if r.std()>0 else 0)
    roll_b = res['bm_rets'].rolling(12).apply(
        lambda r: (r.mean()-rf_m)/r.std()*np.sqrt(12) if r.std()>0 else 0)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=roll_s.index, y=roll_s, name='Strategy',
        line=dict(color='#3fb950', width=1.5)))
    fig3.add_trace(go.Scatter(x=roll_b.index, y=roll_b, name='CSI 300',
        line=dict(color='#f78166', width=1.2)))
    fig3.add_hline(y=0, line_color='#444', line_width=1)
    fig3.add_hline(y=1, line_dash='dash', line_color='#666', line_width=0.8)
    fig3.update_layout(title='Rolling 12-Month Sharpe Ratio',
        template='plotly_dark', height=300,
        margin=dict(l=50, r=20, t=50, b=40), legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig3, use_container_width=True)

# ─── Chart 4: Year-by-year ────────────────────────────────────────────────────
years = sorted(res['strat_rets'].index.year.unique())
yearly_s, yearly_b = [], []
for y in years:
    s = res['strat_rets'][res['strat_rets'].index.year == y]
    b = res['bm_rets'][res['bm_rets'].index.year == y]
    yearly_s.append((1 + s).prod() - 1)
    yearly_b.append((1 + b).prod() - 1)

yearly_df = pd.DataFrame({'Strategy': yearly_s, 'CSI300': yearly_b}, index=years)
alpha_yr  = yearly_df['Strategy'] - yearly_df['CSI300']

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=years, y=yearly_df['Strategy']*100,
    name='Strategy', marker_color=['#3fb950' if v>=0 else '#f85149' for v in yearly_df['Strategy']]))
fig4.add_trace(go.Bar(x=years, y=yearly_df['CSI300']*100,
    name='CSI 300',  marker_color=['rgba(88,166,255,0.6)' if v>=0 else 'rgba(218,54,51,0.6)' for v in yearly_df['CSI300']]))
fig4.add_trace(go.Scatter(x=years, y=alpha_yr*100, name='Alpha',
    mode='markers+lines', line=dict(color='#d2a8ff', width=1.5, dash='dot'),
    marker=dict(size=6)))
fig4.add_hline(y=0, line_color='#444', line_width=1)
beat = (alpha_yr > 0).sum()
fig4.update_layout(
    title=f"Year-by-Year Returns vs 沪深300  (Beat in {beat}/{len(years)} years = {beat/len(years):.0%})",
    yaxis_ticksuffix='%', barmode='group', template='plotly_dark', height=350,
    margin=dict(l=50, r=20, t=50, b=40), legend=dict(x=0.01, y=0.99))
st.plotly_chart(fig4, use_container_width=True)

# ─── Chart 5+6: Sector heatmap + scatter ──────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    # Current top sectors
    last_scores = res['scores'].iloc[-1].dropna().sort_values(ascending=False)
    top_sectors = last_scores.head(top_n)
    colors = ['#3fb950' if i < top_n else '#8b949e'
              for i in range(len(last_scores))]
    fig5 = go.Figure(go.Bar(
        x=last_scores.values[:15], y=last_scores.index[:15],
        orientation='h', marker_color=colors[:15]))
    fig5.update_layout(
        title=f"Current Momentum Scores (Top-{top_n} highlighted)",
        template='plotly_dark', height=350, yaxis_autorange='reversed',
        margin=dict(l=130, r=20, t=50, b=40))
    st.plotly_chart(fig5, use_container_width=True)

with c4:
    # Monthly scatter
    beat_color = ['#3fb950' if s > b else '#f85149'
                  for s, b in zip(res['strat_rets'], res['bm_rets'])]
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=res['bm_rets']*100, y=res['strat_rets']*100,
        mode='markers',
        marker=dict(color=beat_color, size=5, opacity=0.65),
        text=[str(d.date()) for d in res['strat_rets'].index],
        hovertemplate='%{text}<br>CSI300: %{x:.1f}%<br>Strategy: %{y:.1f}%',
        name='Monthly'))
    # y=x line
    lim = max(abs(res['bm_rets'].min()), abs(res['bm_rets'].max())) * 100 * 1.1
    fig6.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim],
        mode='lines', line=dict(color='#444', dash='dot', width=1), name='y=x'))
    # Regression
    m_, c_ = np.polyfit(res['bm_rets'].dropna(), res['strat_rets'].reindex(res['bm_rets'].dropna().index).fillna(0), 1)
    xs = np.linspace(res['bm_rets'].min(), res['bm_rets'].max(), 50)
    fig6.add_trace(go.Scatter(x=xs*100, y=(m_*xs+c_)*100,
        mode='lines', line=dict(color='#d2a8ff', width=1.5),
        name=f'β={m_:.2f}, α={c_*12:.1%}/yr'))
    fig6.update_layout(
        title='Monthly Return Scatter (green = strategy beat)',
        xaxis_title='CSI 300 (%)', yaxis_title='Strategy (%)',
        template='plotly_dark', height=350,
        margin=dict(l=60, r=20, t=50, b=50), legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig6, use_container_width=True)

# ─── Summary table ────────────────────────────────────────────────────────────
st.markdown("### Performance Summary")
summary = pd.DataFrame({
    'Metric': ['CAGR', 'Total Return', 'Ann. Volatility', 'Sharpe Ratio',
               'Max Drawdown', 'Calmar Ratio', 'Alpha/yr', 'Beta', 'Info Ratio'],
    'Strategy': [f"{sm['cagr']:.1%}", f"{sm['total']:.0%}", f"{sm['vol']:.1%}",
                 f"{sm['sharpe']:.2f}", f"{sm['mdd']:.1%}", f"{sm['calmar']:.2f}",
                 f"+{alpha_ann:.1%}", f"{beta:.2f}", f"{ir:.2f}"],
    'CSI 300':  [f"{bm['cagr']:.1%}", f"{bm['total']:.0%}", f"{bm['vol']:.1%}",
                 f"{bm['sharpe']:.2f}", f"{bm['mdd']:.1%}", f"{bm['calmar']:.2f}",
                 "—", "1.00", "—"],
})
st.dataframe(summary, use_container_width=True, hide_index=True,
             column_config={"Metric": st.column_config.TextColumn(width="medium")})

st.caption("⚠️ Past performance does not guarantee future results. "
           "This is for educational/research purposes only.")
