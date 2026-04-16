"""
A-Share Quant Strategy Dashboard
Three strategies vs 沪深300
"""

import os, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import akshare as ak

st.set_page_config(page_title="A-Share Quant", page_icon="📈", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF = 0.025 / 12

# ─── SW sector code→name ──────────────────────────────────────────────────────
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

# ─── Data loaders (cached 24h) ────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def load_index(sym: str) -> pd.Series:
    df = ak.stock_zh_index_daily(symbol=sym)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    s = df['close'].resample('ME').last()
    s.index = s.index.to_period('M').to_timestamp('M')
    return s

@st.cache_data(ttl=86400, show_spinner=False)
def load_sector_prices() -> pd.DataFrame:
    cache = os.path.join(BASE_DIR, 'prices_cache.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True).ffill()
        df.columns = [CODE_TO_NAME.get(c, c) for c in df.columns]
        return df
    # fallback: fetch live
    series = []
    for code, name in CODE_TO_NAME.items():
        try:
            d = ak.index_hist_sw(symbol=code, period='month')
            s = d[['日期','收盘']].copy()
            s['日期'] = pd.to_datetime(s['日期'])
            s = s.set_index('日期').sort_index()['收盘'].rename(name)
            series.append(s)
        except Exception:
            pass
        time.sleep(0.3)
    return pd.concat(series, axis=1).ffill()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def dd(rets): c=(1+rets).cumprod(); return (c-c.cummax())/c.cummax()

def perf(rets, label=''):
    n=len(rets); c=(1+rets).cumprod()
    tot=c.iloc[-1]-1; cagr=(1+tot)**(12/n)-1
    vol=rets.std()*np.sqrt(12)
    sh=(rets.mean()-RF)/rets.std()*np.sqrt(12) if rets.std()>0 else 0
    mdd=dd(rets).min(); cal=cagr/abs(mdd) if mdd else 0
    return dict(label=label,cagr=cagr,tot=tot,vol=vol,sh=sh,mdd=mdd,cal=cal,n=n,cum=c,dd=dd(rets))

def alpha_beta(s, b):
    cov=np.cov(s,b); beta=cov[0,1]/cov[1,1] if cov[1,1]>0 else 0
    return (s.mean()-beta*b.mean())*12, beta

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def run_sector_rotation(prices, bm_rets, top_n, lookbacks, weights,
                        trend_win, tc, risk_adj, start_year):
    """Original risk-adj composite momentum sector rotation."""
    prices = prices[prices.index.year >= start_year].copy()
    bm     = bm_rets[bm_rets.index.year >= start_year].copy()
    mr     = prices.pct_change()
    bm_lvl = (1+bm).cumprod()
    bm_aln = bm_lvl.reindex(prices.index, method='ffill')
    trend  = (bm_aln > bm_aln.rolling(trend_win).mean()).shift(1)
    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for t in range(max(lookbacks)+1, len(prices)):
        row={}
        for col in prices.columns:
            s=0.
            for lb,w in zip(lookbacks,weights):
                p0,p1=prices.iloc[t-lb][col],prices.iloc[t][col]
                if pd.notna(p0) and pd.notna(p1) and p0>0: s+=w*((p1/p0)-1)
            if risk_adj:
                vol=mr.iloc[max(0,t-12):t][col].std()
                if vol>0: s/=vol
            row[col]=s
        scores.iloc[t]=pd.Series(row)
    results=[]; held=set()
    for t in range(max(lookbacks)+2, len(prices)):
        date,prev=prices.index[t],prices.index[t-1]
        in_mkt=bool(trend.iloc[t]) if pd.notna(trend.iloc[t]) else True
        if not in_mkt:
            results.append({'date':date,'ret':0.,'turnover':1. if held else 0.,'in_market':False}); held=set(); continue
        row_s=scores.iloc[t-1].dropna()
        if len(row_s)<top_n:
            results.append({'date':date,'ret':0.,'turnover':0.,'in_market':True}); continue
        new_held=set(row_s.nlargest(top_n).index)
        turn=len(held-new_held)/top_n if held else 1.
        pr=[(prices.loc[date,c]/prices.loc[prev,c])-1 for c in new_held
            if pd.notna(prices.loc[prev,c]) and prices.loc[prev,c]>0]
        net=(np.mean(pr) if pr else 0.)-turn*tc*2
        results.append({'date':date,'ret':net,'turnover':turn,'in_market':True}); held=new_held
    bt=pd.DataFrame(results).set_index('date')
    return bt['ret'], bm.reindex(bt.index).fillna(0), bt, scores


def run_index_rotation(start_year, ma_win=10, tc=0.003):
    """Multi-index momentum rotation: CSI300 / CSI500 / ChiNext / cash."""
    with st.spinner("Loading index prices..."):
        idxs = {n: load_index(s) for n,s in
                [('CSI300','sh000300'),('CSI500','sh000905'),('ChiNext','sz399006')]}
    bm = load_index('sh000300')
    df = pd.concat(idxs, axis=1).dropna()
    df = df[df.index.year >= start_year]
    bm = bm[bm.index.year >= start_year]
    results=[]; held='Cash'
    for t in range(ma_win+1, len(df)):
        date=df.index[t]
        in_mkt=any(df[c].iloc[t-1]>df[c].iloc[max(0,t-ma_win-1):t-1].mean() for c in df.columns)
        if not in_mkt:
            results.append({'date':date,'ret':0.,'held':'Cash'}); held='Cash'; continue
        scores={}
        for col in df.columns:
            s=0.
            for lb,w in [(1,.4),(3,.35),(6,.25)]:
                if t-lb>=0: s+=w*((df[col].iloc[t-1]/df[col].iloc[t-lb-1])-1)
            vol=df[col].pct_change().iloc[max(0,t-6):t-1].std()
            scores[col]=s/vol if vol>0 else s
        best=max(scores,key=scores.get)
        p0,p1=df[best].iloc[t-1],df[best].iloc[t]
        ret=(p1/p0-1) if p0>0 else 0.
        tc_cost=tc*2 if best!=held else 0.
        results.append({'date':date,'ret':ret-tc_cost,'held':best}); held=best
    bt=pd.DataFrame(results).set_index('date')
    bm_r=bm.pct_change().reindex(bt.index).fillna(0)
    return bt['ret'], bm_r, bt


def run_low_vol(prices, bm_rets, top_k=3, vol_window=12, trend_win=10, tc=0.003, start_year=2005):
    """Strategy ④: Low-volatility sector rotation with inverse-vol weighting."""
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
        vols={}
        for col in prices.columns:
            v=mr.iloc[max(0,t-vol_window):t][col].std()
            if pd.notna(v) and v>0: vols[col]=v
        if len(vols)<top_k:
            results.append({'date':date,'ret':0.,'mode':'insufficient'}); continue
        sorted_by_vol=sorted(vols, key=vols.get)
        new_held=set(sorted_by_vol[:top_k])
        inv_vols={c:1/vols[c] for c in new_held}
        total=sum(inv_vols.values())
        w={c:v/total for c,v in inv_vols.items()}
        turn=len(held-new_held)/top_k if held else 1.
        pr=sum(w.get(c,0)*((prices.loc[date,c]/prices.loc[prev,c])-1)
               for c in new_held if pd.notna(prices.loc[prev,c]) and prices.loc[prev,c]>0)
        results.append({'date':date,'ret':pr-turn*tc*2,'mode':'invested'})
        held=new_held
    bt=pd.DataFrame(results).set_index('date')
    return bt['ret'], bm.reindex(bt.index).fillna(0), bt


def run_dual_momentum(prices, bm_rets, top_k, trend_win, tc, start_year):
    """Dual-momentum: absolute + relative, defensive fallback."""
    prices=prices[prices.index.year>=start_year].copy()
    bm=bm_rets[bm_rets.index.year>=start_year].copy()
    mr=prices.pct_change()
    bm_lvl=(1+bm).cumprod()
    bm_aln=bm_lvl.reindex(prices.index,method='ffill')
    timing=(bm_aln>bm_aln.rolling(trend_win).mean()).shift(1)
    results=[]; held=[]
    for t in range(8, len(prices)):
        date,prev=prices.index[t],prices.index[t-1]
        in_mkt=bool(timing.iloc[t]) if pd.notna(timing.iloc[t]) else True
        # compute scores
        scores={}
        for col in prices.columns:
            s=0.
            for lb,w in [(1,.5),(3,.3),(6,.2)]:
                if t-lb>=0:
                    p0,p1=prices.iloc[t-1-lb][col],prices.iloc[t-1][col]
                    if pd.notna(p0) and pd.notna(p1) and p0>0: s+=w*((p1/p0)-1)
            vol=mr.iloc[max(0,t-6):t-1][col].std()
            scores[col]=s/vol if vol>0 else s
        abs_ret={col:(prices.iloc[t-1][col]/prices.iloc[t-2][col]-1)
                  if t>=2 and prices.iloc[t-2][col]>0 else -1 for col in prices.columns}
        if not in_mkt:
            results.append({'date':date,'ret':0.,'mode':'cash'}); held=[]; continue
        qualified={c:sc for c,sc in scores.items() if abs_ret.get(c,-1)>0}
        if len(qualified)>=top_k:
            new_held=set(sorted(qualified,key=qualified.get,reverse=True)[:top_k])
            mode='offensive'
        elif qualified:
            new_held=set(sorted(qualified,key=qualified.get,reverse=True))
            mode='mixed'
        else:
            def_s={c:scores.get(c,-99) for c in DEFENSIVE if c in scores}
            new_held=set(sorted(def_s,key=def_s.get,reverse=True)[:top_k])
            mode='defensive'
        turn=len(set(held)-new_held)/top_k if held else 1.
        pr=[(prices.loc[date,c]/prices.loc[prev,c])-1 for c in new_held
            if pd.notna(prices.loc[prev,c]) and prices.loc[prev,c]>0]
        net=(np.mean(pr) if pr else 0.)-turn*tc*2
        results.append({'date':date,'ret':net,'mode':mode}); held=list(new_held)
    bt=pd.DataFrame(results).set_index('date')
    return bt['ret'], bm.reindex(bt.index).fillna(0), bt

def run_ensemble_blend(prices, bm_rets, trend_win=10, tc=0.003, start_year=2005):
    """Strategy ⑤: Equal-weight ensemble of Strategies ①③④."""
    s1, b1, _, _ = run_sector_rotation(prices, bm_rets, 3, [1,3,6,12], [.4,.3,.2,.1],
                                       trend_win, tc, True, start_year)
    s3, b3, _ = run_low_vol(prices, bm_rets, 3, 12, trend_win, tc, start_year)
    s4, b4, _ = run_dual_momentum(prices, bm_rets, 3, trend_win, tc, start_year)
    idx = s1.index.intersection(s3.index).intersection(s4.index)
    s1, s3, s4 = s1[idx], s3[idx], s4[idx]
    bm = b1.reindex(idx).fillna(0)
    ensemble = (s1 + s3 + s4) / 3
    return ensemble, bm


# ─── Shared chart helpers ─────────────────────────────────────────────────────
COLORS = {'strategy': '#3fb950', 'bm': '#f78166', 'alt': '#58a6ff', 'accent': '#ffd600'}

def cum_chart(s_rets, bm_rets, s_label, title, bt=None):
    sm, bm_ = perf(s_rets), perf(bm_rets)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sm['cum'].index, y=sm['cum'],
        name=f"{s_label}  {sm['cagr']:.1%} CAGR", line=dict(color=COLORS['strategy'],width=2.5)))
    fig.add_trace(go.Scatter(x=bm_['cum'].index, y=bm_['cum'],
        name=f"CSI 300  {bm_['cagr']:.1%} CAGR", line=dict(color=COLORS['bm'],width=1.8)))
    if bt is not None and 'in_market' in bt.columns:
        in_cash=False; cs=None
        for d,row in bt.iterrows():
            if not row.get('in_market',True) and not in_cash: in_cash=True; cs=d
            elif row.get('in_market',True) and in_cash:
                in_cash=False; fig.add_vrect(x0=cs,x1=d,fillcolor='gray',opacity=0.07,line_width=0)
        if in_cash: fig.add_vrect(x0=cs,x1=bt.index[-1],fillcolor='gray',opacity=0.07,line_width=0)
    fig.update_layout(title=title, yaxis_type='log',
        yaxis_tickformat='.1f', yaxis_title='Growth of ¥1',
        height=380, legend=dict(x=.01,y=.99), margin=dict(l=55,r=15,t=45,b=35))
    return fig

def dd_chart(s_rets, bm_rets, title):
    sm, bm_ = perf(s_rets), perf(bm_rets)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sm['dd'].index, y=sm['dd']*100,
        fill='tozeroy', name='Strategy', line=dict(color=COLORS['strategy'],width=1.5),
        fillcolor='rgba(63,185,80,.2)'))
    fig.add_trace(go.Scatter(x=bm_['dd'].index, y=bm_['dd']*100,
        fill='tozeroy', name='CSI 300', line=dict(color=COLORS['bm'],width=1.2),
        fillcolor='rgba(248,113,102,.15)'))
    fig.update_layout(title=title, yaxis_ticksuffix='%', height=300,
        legend=dict(x=.01,y=.05), margin=dict(l=50,r=10,t=45,b=35))
    return fig

def yearly_chart(s_rets, bm_rets, title):
    years=sorted(s_rets.index.year.unique())
    ys=[(1+s_rets[s_rets.index.year==y]).prod()-1 for y in years]
    yb=[(1+bm_rets[bm_rets.index.year==y]).prod()-1 for y in years]
    beat=sum(s>b for s,b in zip(ys,yb))
    fig=go.Figure()
    fig.add_trace(go.Bar(x=years, y=[v*100 for v in ys], name='Strategy',
        marker_color=['#3fb950' if v>=0 else '#f85149' for v in ys]))
    fig.add_trace(go.Bar(x=years, y=[v*100 for v in yb], name='CSI 300',
        marker_color=['rgba(88,166,255,.65)' if v>=0 else 'rgba(218,54,51,.65)' for v in yb]))
    fig.add_trace(go.Scatter(x=years, y=[(s-b)*100 for s,b in zip(ys,yb)],
        name='Alpha', mode='markers+lines',
        line=dict(color=COLORS['accent'],width=1.5,dash='dot'), marker=dict(size=6)))
    fig.add_hline(y=0,line_color='gray',line_width=.8)
    fig.update_layout(title=f'{title}  (Beat {beat}/{len(years)} yrs = {beat/len(years):.0%})',
        yaxis_ticksuffix='%', barmode='group', height=360,
        legend=dict(x=.01,y=.99), margin=dict(l=50,r=15,t=45,b=40))
    return fig

def kpi_row(sm, bm_m, alph, beta):
    cols=st.columns(7)
    def kpi(c,lbl,v,d=None): c.metric(lbl,v,d)
    kpi(cols[0],"CAGR",          f"{sm['cagr']:.1%}",  f"{sm['cagr']-bm_m['cagr']:+.1%} vs CSI300")
    kpi(cols[1],"Total Return",  f"{sm['tot']:.0%}",   f"CSI300: {bm_m['tot']:.0%}")
    kpi(cols[2],"Sharpe",        f"{sm['sh']:.2f}",    f"CSI300: {bm_m['sh']:.2f}")
    kpi(cols[3],"Max Drawdown",  f"{sm['mdd']:.1%}",   f"CSI300: {bm_m['mdd']:.1%}")
    kpi(cols[4],"Alpha/yr",      f"{alph:+.1%}",       None)
    kpi(cols[5],"Beta",          f"{beta:.2f}",         None)
    kpi(cols[6],"Months",        f"{sm['n']}",          None)

def summary_table(sm, bm_m, alph, beta):
    ir_num = None
    st.dataframe(pd.DataFrame({
        'Metric':  ['CAGR','Total Return','Ann. Volatility','Sharpe',
                    'Max Drawdown','Calmar','Alpha/yr','Beta'],
        'Strategy':[f"{sm['cagr']:.1%}",f"{sm['tot']:.0%}",f"{sm['vol']:.1%}",
                    f"{sm['sh']:.2f}",f"{sm['mdd']:.1%}",f"{sm['cal']:.2f}",
                    f"{alph:+.1%}",f"{beta:.2f}"],
        'CSI 300': [f"{bm_m['cagr']:.1%}",f"{bm_m['tot']:.0%}",f"{bm_m['vol']:.1%}",
                    f"{bm_m['sh']:.2f}",f"{bm_m['mdd']:.1%}",f"{bm_m['cal']:.2f}",
                    "—","1.00"],
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 A-Share Quant")
    strategy = st.radio("Strategy", [
        "① Sector Rotation (Original)",
        "② Index Momentum Rotation",
        "③ Dual-Momentum Sector (JoinQuant)",
        "④ Low-Volatility Rotation",
        "⑤ Multi-Strategy Ensemble",
    ])
    st.markdown("---")

    if strategy == "① Sector Rotation (Original)":
        top_n      = st.slider("Top-N sectors", 1, 8, 3)
        trend_win  = st.slider("Trend filter (months)", 3, 24, 10)
        tc_pct     = st.slider("Transaction cost (one-way %)", 0.0, 1.0, 0.3, 0.05)
        start_year = st.slider("Start year", 2005, 2015, 2006)
        risk_adj   = st.checkbox("Risk-adjusted momentum", True)
    elif strategy == "② Index Momentum Rotation":
        ma_win     = st.slider("MA window (months)", 3, 24, 10)
        tc_pct     = st.slider("Transaction cost (one-way %)", 0.0, 1.0, 0.3, 0.05)
        start_year = st.slider("Start year", 2010, 2015, 2011)
    elif strategy == "③ Dual-Momentum Sector (JoinQuant)":
        top_k      = st.slider("Top-K sectors", 1, 6, 3)
        trend_win  = st.slider("Trend filter (months)", 3, 24, 10)
        tc_pct     = st.slider("Transaction cost (one-way %)", 0.0, 1.0, 0.3, 0.05)
        start_year = st.slider("Start year", 2005, 2015, 2005)
    elif strategy == "④ Low-Volatility Rotation":
        top_k      = st.slider("Top-K sectors", 1, 6, 3)
        vol_window = st.slider("Vol lookback (months)", 3, 24, 12)
        trend_win  = st.slider("Trend filter (months)", 3, 24, 10)
        tc_pct     = st.slider("Transaction cost (one-way %)", 0.0, 1.0, 0.3, 0.05)
        start_year = st.slider("Start year", 2005, 2015, 2005)
    else:  # ⑤ Ensemble
        trend_win  = st.slider("Trend filter (months)", 3, 24, 10)
        tc_pct     = st.slider("Transaction cost (one-way %)", 0.0, 1.0, 0.3, 0.05)
        start_year = st.slider("Start year", 2005, 2015, 2006)

    st.markdown("---")
    st.caption("Data: Shenwan L1 indices + CSI broad indices via akshare")
    st.caption("⚠️ Past performance ≠ future results")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ① — Sector Rotation
# ══════════════════════════════════════════════════════════════════════════════
if strategy == "① Sector Rotation (Original)":
    st.title("① Risk-Adjusted Sector Rotation")
    st.caption("28 Shenwan L1 sectors · composite momentum · trend filter → cash")

    with st.spinner("Loading data..."):
        prices = load_sector_prices()
        bm_series = load_index('sh000300')
        bm_rets = bm_series.pct_change().dropna()

    s, b, bt, scores = run_sector_rotation(
        prices, bm_rets, top_n, [1,3,6,12], [.4,.3,.2,.1],
        trend_win, tc_pct/100, risk_adj, start_year)
    sm, bm_m = perf(s), perf(b)
    alph, beta = alpha_beta(s, b)

    st.caption(f"Backtest: **{s.index[0]:%Y-%m}** → **{s.index[-1]:%Y-%m}** · {sm['n']} months · Top-{top_n} sectors · {tc_pct:.2f}% one-way")
    kpi_row(sm, bm_m, alph, beta)
    st.markdown("---")

    st.plotly_chart(cum_chart(s, b, f"Sector Rotation Top-{top_n}",
        "Cumulative Net Return (log scale) · grey = cash", bt), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(dd_chart(s, b, f"Drawdown · Strategy {sm['mdd']:.1%} vs CSI300 {bm_m['mdd']:.1%}"), use_container_width=True)
    with c2:
        last = scores.iloc[-1].dropna().sort_values(ascending=True).tail(15)
        colors = ['#3fb950' if i >= len(last)-top_n else '#8b949e' for i in range(len(last))]
        fig = go.Figure(go.Bar(x=last.values, y=last.index, orientation='h', marker_color=colors))
        fig.update_layout(title=f"Current Scores (Top-{top_n} in green)", height=300,
            margin=dict(l=130,r=15,t=45,b=35))
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(yearly_chart(s, b, "Year-by-Year Returns"), use_container_width=True)
    summary_table(sm, bm_m, alph, beta)

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ② — Index Momentum Rotation
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "② Index Momentum Rotation":
    st.title("② Multi-Index Momentum Rotation")
    st.caption("CSI300 / CSI500 / ChiNext + cash · composite momentum · all-below-MA → cash")

    s, b, bt = run_index_rotation(start_year, ma_win, tc_pct/100)
    sm, bm_m = perf(s), perf(b)
    alph, beta = alpha_beta(s, b)

    st.caption(f"Backtest: **{s.index[0]:%Y-%m}** → **{s.index[-1]:%Y-%m}** · {sm['n']} months · {tc_pct:.2f}% one-way · MA-{ma_win}")
    kpi_row(sm, bm_m, alph, beta)
    st.markdown("---")

    st.plotly_chart(cum_chart(s, b, "Index Rotation",
        "Cumulative Net Return (log scale) · grey = cash"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(dd_chart(s, b, f"Drawdown · Strategy {sm['mdd']:.1%} vs CSI300 {bm_m['mdd']:.1%}"), use_container_width=True)
    with c2:
        alloc = bt['held'].value_counts().reset_index()
        alloc.columns = ['Index', 'Months']
        fig = go.Figure(go.Bar(x=alloc['Index'], y=alloc['Months'],
            marker_color=[COLORS['strategy'] if i==0 else COLORS['alt'] if i==1 else '#e3b341' if i==2 else '#8b949e'
                          for i in range(len(alloc))]))
        fig.update_layout(title="Allocation History (months held)", height=300,
            margin=dict(l=50,r=15,t=45,b=35))
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(yearly_chart(s, b, "Year-by-Year Returns"), use_container_width=True)
    summary_table(sm, bm_m, alph, beta)

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ③ — Dual-Momentum Sector
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "③ Dual-Momentum Sector (JoinQuant)":
    st.title("③ Dual-Momentum Sector Rotation")
    st.caption("28 Shenwan L1 sectors · absolute + relative momentum · defensive fallback (Banking/Utilities/F&B)")

    with st.spinner("Loading data..."):
        prices = load_sector_prices()
        bm_series = load_index('sh000300')
        bm_rets = bm_series.pct_change().dropna()

    s, b, bt = run_dual_momentum(prices, bm_rets, top_k, trend_win, tc_pct/100, start_year)
    sm, bm_m = perf(s), perf(b)
    alph, beta = alpha_beta(s, b)

    st.caption(f"Backtest: **{s.index[0]:%Y-%m}** → **{s.index[-1]:%Y-%m}** · {sm['n']} months · Top-{top_k} sectors · {tc_pct:.2f}% one-way")
    kpi_row(sm, bm_m, alph, beta)
    st.markdown("---")

    st.plotly_chart(cum_chart(s, b, f"Dual-Mom Sector Top-{top_k}",
        "Cumulative Net Return (log scale) · grey = cash"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(dd_chart(s, b, f"Drawdown · Strategy {sm['mdd']:.1%} vs CSI300 {bm_m['mdd']:.1%}"), use_container_width=True)
    with c2:
        if 'mode' in bt.columns:
            mc = bt['mode'].value_counts()
            fig = go.Figure(go.Pie(labels=mc.index, values=mc.values, hole=.45,
                marker_colors=[COLORS['strategy'],'#58a6ff','#e3b341','#8b949e']))
            fig.update_layout(title="Regime Breakdown (months)", height=300,
                margin=dict(l=20,r=20,t=45,b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(yearly_chart(s, b, "Year-by-Year Returns"), use_container_width=True)
    summary_table(sm, bm_m, alph, beta)

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ④ — Low-Volatility Rotation
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "④ Low-Volatility Rotation":
    st.title("④ Low-Volatility Sector Rotation")
    st.caption("28 Shenwan L1 sectors · lowest trailing-vol sectors · inverse-vol weighted · trend filter → cash")

    with st.spinner("Loading data..."):
        prices = load_sector_prices()
        bm_series = load_index('sh000300')
        bm_rets = bm_series.pct_change().dropna()

    s, b, bt = run_low_vol(prices, bm_rets, top_k, vol_window, trend_win, tc_pct/100, start_year)
    sm, bm_m = perf(s), perf(b)
    alph, beta = alpha_beta(s, b)

    st.caption(f"Backtest: **{s.index[0]:%Y-%m}** → **{s.index[-1]:%Y-%m}** · {sm['n']} months · Top-{top_k} sectors · {tc_pct:.2f}% one-way")
    kpi_row(sm, bm_m, alph, beta)
    st.markdown("---")

    st.plotly_chart(cum_chart(s, b, f"Low-Vol Rotation Top-{top_k}",
        "Cumulative Net Return (log scale) · grey = cash"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(dd_chart(s, b, f"Drawdown · Strategy {sm['mdd']:.1%} vs CSI300 {bm_m['mdd']:.1%}"), use_container_width=True)
    with c2:
        if 'mode' in bt.columns:
            mc = bt['mode'].value_counts()
            fig = go.Figure(go.Pie(labels=mc.index, values=mc.values, hole=.45,
                marker_colors=[COLORS['strategy'],'#58a6ff','#e3b341','#8b949e']))
            fig.update_layout(title="Regime Breakdown (months)", height=300,
                margin=dict(l=20,r=20,t=45,b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(yearly_chart(s, b, "Year-by-Year Returns"), use_container_width=True)
    summary_table(sm, bm_m, alph, beta)

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ⑤ — Multi-Strategy Ensemble
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("⑤ Multi-Strategy Ensemble")
    st.caption("Equal-weight blend of ① Sector Rotation + ④ Low-Vol + ③ Dual-Momentum · diversifies strategy-specific risk")

    with st.spinner("Loading data..."):
        prices = load_sector_prices()
        bm_series = load_index('sh000300')
        bm_rets = bm_series.pct_change().dropna()

    s, b = run_ensemble_blend(prices, bm_rets, trend_win, tc_pct/100, start_year)
    sm, bm_m = perf(s), perf(b)
    alph, beta = alpha_beta(s, b)

    st.caption(f"Backtest: **{s.index[0]:%Y-%m}** → **{s.index[-1]:%Y-%m}** · {sm['n']} months · {tc_pct:.2f}% one-way")
    kpi_row(sm, bm_m, alph, beta)
    st.markdown("---")

    st.plotly_chart(cum_chart(s, b, "Multi-Strategy Ensemble",
        "Cumulative Net Return (log scale)"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(dd_chart(s, b, f"Drawdown · Ensemble {sm['mdd']:.1%} vs CSI300 {bm_m['mdd']:.1%}"), use_container_width=True)
    with c2:
        labels=['① Sector Rotation','④ Low-Vol','③ Dual-Momentum']
        fig = go.Figure(go.Pie(labels=labels, values=[1,1,1], hole=.45,
            marker_colors=[COLORS['strategy'], COLORS['alt'], COLORS['accent']]))
        fig.update_layout(title="Equal-Weight Composition", height=300,
            margin=dict(l=20,r=20,t=45,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(yearly_chart(s, b, "Year-by-Year Returns"), use_container_width=True)
    summary_table(sm, bm_m, alph, beta)

st.caption("Source: Shenwan L1 sector indices + CSI broad indices via akshare  ·  "
           "github.com/LangxiaoXie/quant-strategy")
