"""
A-Share Sector Rotation Dashboard
Loads from bundled CSV data; optionally refreshes via akshare.
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="A-Share Sector Rotation",
    page_icon="📈",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────
SW_INDUSTRIES = {
    '801010': 'Agriculture',     '801020': 'Mining',
    '801030': 'Chemicals',       '801040': 'Steel',
    '801050': 'Non-Ferrous',     '801080': 'Electronics',
    '801110': 'Home Appliances', '801120': 'Food & Beverage',
    '801130': 'Textiles',        '801140': 'Light Mfg',
    '801150': 'Pharma',          '801160': 'Utilities',
    '801170': 'Transport',       '801180': 'Real Estate',
    '801200': 'Commerce',        '801210': 'Leisure',
    '801230': 'Conglomerates',   '801710': 'Construction Mtl',
    '801720': 'Construction Dec','801730': 'Electrical Equip',
    '801740': 'Defense',         '801750': 'IT & Computer',
    '801760': 'Media',           '801770': 'Telecom',
    '801780': 'Banking',         '801790': 'Non-Bank Finance',
    '801880': 'Automotive',      '801890': 'Machinery',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices() -> pd.DataFrame:
    """Load sector prices from CSV, or fetch from akshare if unavailable."""
    cache = os.path.join(BASE_DIR, 'prices_cache.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df.ffill()
    # Fallback: try akshare
    try:
        import akshare as ak
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
    except Exception as e:
        st.error(f"Could not load sector data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark() -> pd.Series:
    """Load CSI300 from monthly_returns CSV or akshare."""
    mr = os.path.join(BASE_DIR, 'monthly_returns.csv')
    if os.path.exists(mr):
        df = pd.read_csv(mr, index_col=0, parse_dates=True)
        if 'csi300' in df.columns:
            # Reconstruct cumulative from returns, then return the returns series
            return df['csi300'].dropna()
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol='sh000300')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        monthly = df['close'].resample('ME').last()
        monthly.index = monthly.index.to_period('M').to_timestamp('M')
        return monthly.pct_change().dropna().rename('csi300')
    except Exception as e:
        st.error(f"Could not load benchmark: {e}")
        return pd.Series(dtype=float)


# ─── Strategy ─────────────────────────────────────────────────────────────────
def run_strategy(prices, bm_rets, top_n, trend_window, tc, risk_adj, start_year):
    prices = prices[prices.index.year >= start_year].copy()
    bm     = bm_rets[bm_rets.index.year >= start_year].copy()

    monthly_rets = prices.pct_change()

    # Reconstruct benchmark levels for trend filter
    bm_level = (1 + bm).cumprod()
    bm_aligned = bm_level.reindex(prices.index, method='ffill')
    ma = bm_aligned.rolling(trend_window).mean()
    trend = (bm_aligned > ma).shift(1)

    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    lookbacks = [1, 3, 6, 12]
    weights   = [0.4, 0.3, 0.2, 0.1]
    max_lb = max(lookbacks)

    for t in range(max_lb + 1, len(prices)):
        row = {}
        for col in prices.columns:
            s = 0.0
            for lb, w in zip(lookbacks, weights):
                p0 = prices.iloc[t - lb][col]
                p1 = prices.iloc[t][col]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    s += w * ((p1 / p0) - 1)
            if risk_adj:
                vol = monthly_rets.iloc[max(0, t-12):t][col].std()
                if vol > 0 and not np.isnan(vol):
                    s /= vol
            row[col] = s
        scores.iloc[t] = pd.Series(row)

    results = []
    held = set()
    for t in range(max_lb + 2, len(prices)):
        date = prices.index[t]
        prev = prices.index[t - 1]
        in_mkt = bool(trend.iloc[t]) if pd.notna(trend.iloc[t]) else True

        if not in_mkt:
            turnover = 1.0 if held else 0.0
            held = set()
            results.append({'date': date, 'ret': 0.0, 'turnover': turnover,
                            'in_market': False, 'held': []})
            continue

        row_s = scores.iloc[t - 1].dropna()
        if len(row_s) < top_n:
            results.append({'date': date, 'ret': 0.0, 'turnover': 0.0,
                            'in_market': True, 'held': list(held)})
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
    bm_aligned_rets = bm.reindex(strat_rets.index).fillna(0)
    return strat_rets, bm_aligned_rets, bt, scores


def calc_metrics(rets):
    rf = 0.025 / 12
    n  = len(rets)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr  = (1 + total) ** (12 / n) - 1
    vol   = rets.std() * np.sqrt(12)
    sharpe = (rets.mean() - rf) / rets.std() * np.sqrt(12) if rets.std() > 0 else 0
    dd = (cum - cum.cummax()) / cum.cummax()
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return dict(cagr=cagr, total=total, vol=vol, sharpe=sharpe,
                mdd=mdd, calmar=calmar, n=n, cum=cum, dd=dd)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")
    top_n      = st.slider("Top-N sectors to hold", 1, 8, 3)
    trend_win  = st.slider("Trend filter window (months)", 3, 24, 10)
    tc_pct     = st.slider("Transaction cost (one-way %)", 0.0, 1.0, 0.3, 0.05)
    start_year = st.slider("Start year", 2005, 2015, 2006)
    risk_adj   = st.checkbox("Risk-adjusted momentum", value=True)
    st.markdown("---")
    st.caption("Lookback: 1m×0.4 + 3m×0.3 + 6m×0.2 + 12m×0.1")
    st.caption("Trend filter: invest only when CSI 300 > N-month MA")
    st.caption("Data: Shenwan Level-1 indices via akshare")

# ─── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    prices  = load_prices()
    bm_rets = load_benchmark()

if prices.empty or bm_rets.empty:
    st.error("Failed to load data. Please check the data files or network connection.")
    st.stop()

# ─── Run strategy ─────────────────────────────────────────────────────────────
strat_rets, bm_r, bt, scores = run_strategy(
    prices, bm_rets, top_n, trend_win, tc_pct/100, risk_adj, start_year)

sm = calc_metrics(strat_rets)
bm_m = calc_metrics(bm_r)

cov = np.cov(strat_rets, bm_r)
beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
alpha_ann = (strat_rets.mean() - beta * bm_r.mean()) * 12
excess = strat_rets - bm_r
ir = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📈 A-Share Sector Rotation Strategy")
st.caption(
    f"Backtest: **{strat_rets.index[0].strftime('%Y-%m')}** → "
    f"**{strat_rets.index[-1].strftime('%Y-%m')}**  ·  "
    f"{sm['n']} months  ·  Top-{top_n} Shenwan L1 sectors  ·  "
    f"Monthly rebalance  ·  {tc_pct:.2f}% one-way cost"
)

# ─── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
def kpi(col, label, val, delta=None):
    col.metric(label, val, delta)

kpi(c1, "CAGR",         f"{sm['cagr']:.1%}",    f"{sm['cagr']-bm_m['cagr']:+.1%} vs CSI300")
kpi(c2, "Total Return", f"{sm['total']:.0%}",   f"CSI300: {bm_m['total']:.0%}")
kpi(c3, "Sharpe",       f"{sm['sharpe']:.2f}",  f"CSI300: {bm_m['sharpe']:.2f}")
kpi(c4, "Max Drawdown", f"{sm['mdd']:.1%}",     f"CSI300: {bm_m['mdd']:.1%}")
kpi(c5, "Alpha/yr",     f"{alpha_ann:+.1%}",    None)
kpi(c6, "Beta",         f"{beta:.2f}",           None)
kpi(c7, "Info Ratio",   f"{ir:.2f}",             None)

st.markdown("---")

# ─── Chart 1: Cumulative returns ──────────────────────────────────────────────
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=sm['cum'].index, y=sm['cum'].values,
    name=f"Strategy  CAGR {sm['cagr']:.1%}  Total {sm['total']:.0%}",
    line=dict(color='#00c853', width=2.5)))
fig1.add_trace(go.Scatter(
    x=bm_m['cum'].index, y=bm_m['cum'].values,
    name=f"CSI 300   CAGR {bm_m['cagr']:.1%}  Total {bm_m['total']:.0%}",
    line=dict(color='#ff5252', width=1.8)))

# Shade cash periods
in_cash = False; cash_start = None
for d, row in bt.iterrows():
    if not row['in_market'] and not in_cash:
        in_cash = True; cash_start = d
    elif row['in_market'] and in_cash:
        in_cash = False
        fig1.add_vrect(x0=cash_start, x1=d, fillcolor='gray', opacity=0.08, line_width=0)
if in_cash:
    fig1.add_vrect(x0=cash_start, x1=bt.index[-1], fillcolor='gray', opacity=0.08, line_width=0)

fig1.update_layout(
    title="Cumulative Net Return (log scale) · grey bands = in cash (trend filter)",
    yaxis_type="log",
    yaxis_tickformat=".1f",
    yaxis_title="Growth of ¥1",
    height=400,
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=60, r=20, t=50, b=40),
)
st.plotly_chart(fig1, use_container_width=True)

# ─── Row 2: Drawdown + Rolling Sharpe ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sm['dd'].index, y=sm['dd'].values * 100,
        fill='tozeroy', name='Strategy',
        line=dict(color='#00c853', width=1.5),
        fillcolor='rgba(0,200,83,0.2)'))
    fig2.add_trace(go.Scatter(
        x=bm_m['dd'].index, y=bm_m['dd'].values * 100,
        fill='tozeroy', name='CSI 300',
        line=dict(color='#ff5252', width=1.2),
        fillcolor='rgba(255,82,82,0.15)'))
    fig2.update_layout(
        title=f"Drawdown  (Strategy {sm['mdd']:.1%}  CSI300 {bm_m['mdd']:.1%})",
        yaxis_ticksuffix='%', height=320,
        legend=dict(x=0.01, y=0.05),
        margin=dict(l=50, r=10, t=50, b=40))
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    rf_m = 0.025 / 12
    roll_s = strat_rets.rolling(12).apply(
        lambda r: (r.mean()-rf_m)/r.std()*np.sqrt(12) if r.std()>0 else 0)
    roll_b = bm_r.rolling(12).apply(
        lambda r: (r.mean()-rf_m)/r.std()*np.sqrt(12) if r.std()>0 else 0)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=roll_s.index, y=roll_s,
        name='Strategy', line=dict(color='#00c853', width=1.5)))
    fig3.add_trace(go.Scatter(x=roll_b.index, y=roll_b,
        name='CSI 300',  line=dict(color='#ff5252', width=1.2)))
    fig3.add_hline(y=0, line_color='gray', line_width=1)
    fig3.add_hline(y=1, line_dash='dash', line_color='gray', line_width=0.8)
    fig3.update_layout(
        title='Rolling 12-Month Sharpe Ratio',
        height=320,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=50, r=10, t=50, b=40))
    st.plotly_chart(fig3, use_container_width=True)

# ─── Chart 3: Year-by-year ────────────────────────────────────────────────────
years = sorted(strat_rets.index.year.unique())
yearly_s = [(1 + strat_rets[strat_rets.index.year == y]).prod() - 1 for y in years]
yearly_b = [(1 + bm_r[bm_r.index.year == y]).prod() - 1 for y in years]
alpha_yr  = [s - b for s, b in zip(yearly_s, yearly_b)]
beat = sum(a > 0 for a in alpha_yr)

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=years, y=[v*100 for v in yearly_s], name='Strategy',
    marker_color=['#00c853' if v >= 0 else '#d32f2f' for v in yearly_s]))
fig4.add_trace(go.Bar(x=years, y=[v*100 for v in yearly_b], name='CSI 300',
    marker_color=['rgba(33,150,243,0.6)' if v >= 0 else 'rgba(183,28,28,0.6)' for v in yearly_b]))
fig4.add_trace(go.Scatter(x=years, y=[v*100 for v in alpha_yr], name='Alpha',
    mode='markers+lines', line=dict(color='#ffd600', width=1.5, dash='dot'),
    marker=dict(size=6)))
fig4.add_hline(y=0, line_color='gray', line_width=1)
fig4.update_layout(
    title=f"Year-by-Year Returns vs CSI 300  (Beat in {beat}/{len(years)} years = {beat/len(years):.0%})",
    yaxis_ticksuffix='%', barmode='group', height=380,
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=50, r=20, t=50, b=40))
st.plotly_chart(fig4, use_container_width=True)

# ─── Row 4: Sector scores + Scatter ───────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    last_scores = scores.iloc[-1].dropna().sort_values(ascending=True).tail(15)
    colors = ['#00c853' if i >= len(last_scores) - top_n else '#90a4ae'
              for i in range(len(last_scores))]
    fig5 = go.Figure(go.Bar(
        x=last_scores.values, y=last_scores.index,
        orientation='h', marker_color=colors))
    fig5.update_layout(
        title=f"Current Momentum Scores (Top-{top_n} in green)",
        height=380, margin=dict(l=130, r=20, t=50, b=40))
    st.plotly_chart(fig5, use_container_width=True)

with col4:
    beat_colors = ['#00c853' if s > b else '#ff5252'
                   for s, b in zip(strat_rets, bm_r)]
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=bm_r*100, y=strat_rets*100, mode='markers',
        marker=dict(color=beat_colors, size=5, opacity=0.65),
        text=[str(d.date()) for d in strat_rets.index],
        hovertemplate='%{text}<br>CSI300: %{x:.1f}%<br>Strategy: %{y:.1f}%',
        name='Monthly'))
    lim = max(abs(bm_r.min()), abs(bm_r.max())) * 100 * 1.1
    fig6.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines',
        line=dict(color='gray', dash='dot', width=1), name='y=x'))
    m_, c_ = np.polyfit(bm_r, strat_rets, 1)
    xs = np.linspace(bm_r.min(), bm_r.max(), 50)
    fig6.add_trace(go.Scatter(x=xs*100, y=(m_*xs+c_)*100, mode='lines',
        line=dict(color='#ffd600', width=1.5),
        name=f'β={m_:.2f}  α={c_*12:.1%}/yr'))
    fig6.update_layout(
        title='Monthly Return Scatter (green = strategy beat)',
        xaxis_title='CSI 300 Monthly Return (%)',
        yaxis_title='Strategy Monthly Return (%)',
        height=380, legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=20, t=50, b=50))
    st.plotly_chart(fig6, use_container_width=True)

# ─── Summary table ────────────────────────────────────────────────────────────
st.markdown("### Performance Summary")
st.dataframe(pd.DataFrame({
    'Metric':   ['CAGR', 'Total Return', 'Ann. Volatility', 'Sharpe Ratio',
                 'Max Drawdown', 'Calmar Ratio', 'Alpha/yr', 'Beta', 'Info Ratio'],
    'Strategy': [f"{sm['cagr']:.1%}", f"{sm['total']:.0%}", f"{sm['vol']:.1%}",
                 f"{sm['sharpe']:.2f}", f"{sm['mdd']:.1%}", f"{sm['calmar']:.2f}",
                 f"{alpha_ann:+.1%}", f"{beta:.2f}", f"{ir:.2f}"],
    'CSI 300':  [f"{bm_m['cagr']:.1%}", f"{bm_m['total']:.0%}", f"{bm_m['vol']:.1%}",
                 f"{bm_m['sharpe']:.2f}", f"{bm_m['mdd']:.1%}", f"{bm_m['calmar']:.2f}",
                 "—", "1.00", "—"],
}), use_container_width=True, hide_index=True)

st.caption("⚠️ Past performance does not guarantee future results. For research purposes only.")
