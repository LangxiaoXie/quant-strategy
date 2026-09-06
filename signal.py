"""
signal.py — Monthly three-factor sector rotation signal + WeChat push via Server酱.

Usage:
    python3 signal.py           # runs only on last trading day of month
    python3 signal.py --force   # bypass date check, always compute + push

Config:
    SERVERCHAN_KEY env var  OR  ~/.quant_config file containing:
        SERVERCHAN_KEY=SCT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    For multiple recipients (e.g. yourself + family), comma-separate keys:
        SERVERCHAN_KEY=SCT_mykey,SCT_momkey
"""

import argparse
import calendar
import json
import os
import sys
import time
from datetime import date, timedelta

import akshare as ak
import numpy as np
import pandas as pd
import requests

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PRICES_CACHE  = os.path.join(BASE_DIR, "prices_cache.csv")
HOLDINGS_FILE = os.path.join(BASE_DIR, "holdings.json")

# ─── Strategy ⑥ default params (match app.py sidebar defaults) ──────────────
BIAS_N    = 20
SLOPE_N   = 20
MOM_DAY   = 12
EFF_LB    = 12
W_BIAS    = 0.3
W_SLOPE   = 0.3
W_EFF     = 0.4
THRESHOLD = 1.5
TREND_WIN = 10

# ─── SW sector code → name ────────────────────────────────────────────────────
CODE_TO_NAME = {
    "801010": "Agriculture",      "801020": "Mining",
    "801030": "Chemicals",        "801040": "Steel",
    "801050": "Non-Ferrous",      "801080": "Electronics",
    "801110": "Home Appliances",  "801120": "Food & Beverage",
    "801130": "Textiles",         "801140": "Light Mfg",
    "801150": "Pharma",           "801160": "Utilities",
    "801170": "Transport",        "801180": "Real Estate",
    "801200": "Commerce",         "801210": "Leisure",
    "801230": "Conglomerates",    "801710": "Construction Mtl",
    "801720": "Construction Dec", "801730": "Electrical Equip",
    "801740": "Defense",          "801750": "IT & Computer",
    "801760": "Media",            "801770": "Telecom",
    "801780": "Banking",          "801790": "Non-Bank Finance",
    "801880": "Automotive",       "801890": "Machinery",
}

# ─── Sector → most-liquid ETF mapping ────────────────────────────────────────
# Tickers are Shanghai/Shenzhen fund codes tradeable via normal brokerage.
SECTOR_ETF = {
    "Agriculture":      "159825",  # 农业ETF
    "Mining":           "601899",  # 紫金矿业 (no pure mining ETF; use proxy)
    "Chemicals":        "516020",  # 化工ETF
    "Steel":            "516150",  # 钢铁ETF
    "Non-Ferrous":      "512400",  # 有色ETF
    "Electronics":      "515260",  # 电子ETF (科技龙头)
    "Home Appliances":  "159996",  # 家电ETF
    "Food & Beverage":  "159843",  # 食品饮料ETF
    "Textiles":         "512990",  # 纺织服装ETF
    "Light Mfg":        "159766",  # 轻工制造ETF
    "Pharma":           "512010",  # 医药ETF
    "Utilities":        "159611",  # 公用事业ETF
    "Transport":        "516110",  # 交通运输ETF
    "Real Estate":      "512200",  # 房地产ETF
    "Commerce":         "516030",  # 商贸零售ETF
    "Leisure":          "159766",  # 休闲服务 (use light mfg as fallback)
    "Conglomerates":    "510880",  # 红利ETF (conglomerates proxy)
    "Construction Mtl": "159745",  # 建材ETF
    "Construction Dec": "516220",  # 装饰建材ETF
    "Electrical Equip": "515050",  # 新能源ETF (electrical equip proxy)
    "Defense":          "512660",  # 国防军工ETF
    "IT & Computer":    "512720",  # 计算机ETF
    "Media":            "512980",  # 传媒ETF
    "Telecom":          "515880",  # 通信ETF
    "Banking":          "512800",  # 银行ETF
    "Non-Bank Finance": "512070",  # 证券ETF
    "Automotive":       "516110",  # 汽车ETF proxy
    "Machinery":        "516620",  # 机械ETF
}

# ─── Major A-share public holidays (month, day) ──────────────────────────────
# Covers fixed-date holidays; Spring Festival moves yearly so we use a rolling
# 5-year hard-coded list plus a fallback "close enough" heuristic.
_FIXED_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (5, 1),   # Labour Day
    (10, 1),  # National Day
    (10, 2),
    (10, 3),
    (10, 4),
    (10, 5),
    (10, 6),
    (10, 7),
}
# Spring Festival approximate block (Jan 20 – Feb 10 bracket)
# We mark Feb 1-10 as potentially closed; conservative — only matters for
# last-trading-day detection in January/February.
_SPRING_FESTIVAL_APPROX = {(1, d) for d in range(20, 32)} | {(2, d) for d in range(1, 11)}


def _is_holiday(d: date) -> bool:
    """Return True if d is a known public holiday (simplified)."""
    return (d.month, d.day) in _FIXED_HOLIDAYS or (d.month, d.day) in _SPRING_FESTIVAL_APPROX


def _is_trading_day(d: date) -> bool:
    """Return True if d is a weekday and not a known holiday."""
    return d.weekday() < 5 and not _is_holiday(d)


def last_trading_day_of_month(ref: date) -> date:
    """Return the last trading day of the month containing ref."""
    last_day = date(ref.year, ref.month, calendar.monthrange(ref.year, ref.month)[1])
    candidate = last_day
    while not _is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def is_last_trading_day(ref: date | None = None) -> bool:
    """Return True if ref (default today) is the last trading day of its month."""
    today = ref or date.today()
    return today == last_trading_day_of_month(today)


# ─── Scoring functions (copied verbatim from app.py) ─────────────────────────

def _bias_score(close, bias_n, mom_day):
    """乖离动量: linear trend of price-deviation-from-MA."""
    if len(close) < bias_n + mom_day:
        return np.nan
    bias = close / close.rolling(bias_n).mean()
    recent = bias.iloc[-mom_day:].values
    if np.any(np.isnan(recent)) or recent[0] == 0:
        return np.nan
    y = recent / recent[0]
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return slope * 10000


def _slope_score(close, slope_n):
    """斜率动量: regression slope × R² on normalised prices."""
    if len(close) < slope_n:
        return np.nan
    p = close.iloc[-slope_n:].values.astype(float)
    if np.any(np.isnan(p)) or p[0] == 0:
        return np.nan
    norm = p / p[0]
    x = np.arange(1, slope_n + 1, dtype=float)
    slope, intercept = np.polyfit(x, norm, 1)
    ss_res = np.sum((norm - (slope * x + intercept)) ** 2)
    ss_tot = np.sum((norm - norm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return 10000 * slope * r2


def _efficiency_score(close, lookback):
    """效率动量: log-return × efficiency ratio (net / total path)."""
    if len(close) < lookback:
        return np.nan
    lp = np.log(close.iloc[-lookback:].values)
    if lp[0] == 0:
        return np.nan
    mom = 100 * (lp[-1] - lp[0])
    direction = abs(lp[-1] - lp[0])
    total_path = np.abs(np.diff(lp)).sum()
    er = direction / total_path if total_path > 0 else 0
    return mom * er


def _zscore(vals):
    s = pd.Series(vals, dtype=float)
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_sector_prices() -> pd.DataFrame:
    """Load sector prices from cache CSV, renaming codes to sector names."""
    if not os.path.exists(PRICES_CACHE):
        raise FileNotFoundError(f"prices_cache.csv not found at {PRICES_CACHE}")
    df = pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True).ffill()
    df.columns = [CODE_TO_NAME.get(c, c) for c in df.columns]
    return df


def load_csi300_monthly() -> pd.Series:
    """Fetch CSI300 daily prices from akshare and resample to month-end."""
    print("Fetching CSI300 from akshare…")
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    s = df["close"].resample("ME").last()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s


# ─── Core signal computation ──────────────────────────────────────────────────

def compute_scores(prices: pd.DataFrame) -> pd.Series:
    """
    Compute three-factor composite score for each sector using all available
    history up to the latest row. Returns a Series indexed by sector name.
    """
    t = len(prices) - 1  # use the latest available row
    raw = {"bias": {}, "slope": {}, "eff": {}}
    for col in prices.columns:
        c = prices[col].iloc[: t + 1]
        raw["bias"][col]  = _bias_score(c, BIAS_N, MOM_DAY)
        raw["slope"][col] = _slope_score(c, SLOPE_N)
        raw["eff"][col]   = _efficiency_score(c, EFF_LB)

    cols = list(prices.columns)
    zb = _zscore(list(raw["bias"].values()))
    zs = _zscore(list(raw["slope"].values()))
    ze = _zscore(list(raw["eff"].values()))

    composite = {}
    for i, col in enumerate(cols):
        if pd.notna(zb.iloc[i]) and pd.notna(zs.iloc[i]) and pd.notna(ze.iloc[i]):
            composite[col] = W_BIAS * zb.iloc[i] + W_SLOPE * zs.iloc[i] + W_EFF * ze.iloc[i]
    return pd.Series(composite).sort_values(ascending=False)


def check_trend(bm_series: pd.Series) -> bool:
    """Return True (in-market) if latest CSI300 is above its TREND_WIN-month MA."""
    ma = bm_series.rolling(TREND_WIN).mean()
    return bool(bm_series.iloc[-1] > ma.iloc[-1])


# ─── Holdings persistence ──────────────────────────────────────────────────────

def load_holdings() -> dict:
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE) as f:
            return json.load(f)
    return {}


def save_holdings(sector: str, etf: str, since: str):
    data = {"sector": sector, "etf": etf, "since": since}
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"holdings.json updated → {data}")


# ─── WeChat push via Server酱 (sct.ftqq.com) ─────────────────────────────────

def _read_config(key: str) -> str:
    """Read a key from env or ~/.quant_config file."""
    val = os.environ.get(key, "")
    if val:
        return val
    cfg = os.path.expanduser("~/.quant_config")
    if os.path.exists(cfg):
        with open(cfg) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip()
    return ""


def push_wechat(title: str, content: str) -> bool:
    """Send a message via Server酱 to every key in SERVERCHAN_KEY (comma-separated
    for multiple recipients, e.g. 'SCT_mine,SCT_mom'). Returns True if any succeeded."""
    keys = [k.strip() for k in _read_config("SERVERCHAN_KEY").split(",") if k.strip()]
    if not keys:
        print("WARNING: SERVERCHAN_KEY not set — skipping WeChat push.")
        print(f"  Title  : {title}")
        print(f"  Content: {content}")
        return False

    any_success = False
    for key in keys:
        url = f"https://sctapi.ftqq.com/{key}.send"
        try:
            r = requests.post(url, data={"title": title, "desp": content}, timeout=10)
            data = r.json()
            if data.get("code") == 0:
                print(f"WeChat push sent to {key[:10]}...: {title}")
                any_success = True
            else:
                print(f"Server酱 error for {key[:10]}...: {data}")
        except Exception as e:
            print(f"Server酱 request failed for {key[:10]}...: {e}")
    return any_success


# ─── Main logic ───────────────────────────────────────────────────────────────

def run(force: bool = False):
    today = date.today()

    # ── 1. Date gate ──────────────────────────────────────────────────────────
    if not force and not is_last_trading_day(today):
        ltd = last_trading_day_of_month(today)
        print(f"Today ({today}) is not the last trading day of the month ({ltd}). Exiting.")
        sys.exit(0)

    print(f"=== signal.py running on {today} {'[FORCED]' if force else ''} ===")

    # ── 2. Load data ──────────────────────────────────────────────────────────
    prices = load_sector_prices()
    try:
        bm = load_csi300_monthly()
    except Exception as e:
        print(f"WARNING: Could not fetch CSI300 ({e}). Assuming in-market.")
        bm = None

    # ── 3. Trend filter ───────────────────────────────────────────────────────
    in_market = True
    if bm is not None:
        in_market = check_trend(bm)
        print(f"Trend filter: {'IN MARKET' if in_market else 'OUT (cash)'}")

    # ── 4. Load current holdings ──────────────────────────────────────────────
    holdings = load_holdings()
    current_sector = holdings.get("sector", "")
    current_etf    = holdings.get("etf", "")
    held_since     = holdings.get("since", "")

    # ── 5. Compute scores ─────────────────────────────────────────────────────
    scores = compute_scores(prices)
    top_sector = scores.index[0]
    top_score  = scores.iloc[0]

    print(f"\nTop-5 composite scores:")
    for sec, sc in scores.head(5).items():
        marker = " ◀ current" if sec == current_sector else ""
        print(f"  {sc:+.4f}  {sec}{marker}")

    # ── 6. Decision ───────────────────────────────────────────────────────────
    action = "HOLD"
    new_sector = current_sector
    new_etf    = current_etf

    if not in_market:
        # Trend says go to cash
        if current_sector:
            action = "SELL_ALL"
            new_sector = ""
            new_etf    = ""
        else:
            action = "CASH"  # already in cash

    elif not current_sector:
        # No position yet → buy
        action = "BUY"
        new_sector = top_sector
        new_etf    = SECTOR_ETF.get(top_sector, "N/A")

    else:
        current_score = scores.get(current_sector, -np.inf)
        print(f"\nThreshold check: {top_score:.4f} vs {current_score:.4f} × {THRESHOLD} = {current_score * THRESHOLD:.4f}")
        if top_sector != current_sector and top_score > current_score * THRESHOLD:
            action = "SWITCH"
            new_sector = top_sector
            new_etf    = SECTOR_ETF.get(top_sector, "N/A")
        else:
            action = "HOLD"

    print(f"\nDecision: {action}")

    # ── 7. Format message ─────────────────────────────────────────────────────
    date_str = today.strftime("%Y-%m-%d")

    if action == "HOLD":
        title = f"[量化信号] {date_str} — 持仓不变"
        content = (
            f"月末信号 · {date_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"操作：持仓不变 (HOLD)\n\n"
            f"当前持仓：{current_sector} / ETF {current_etf}\n"
            f"持有自：{held_since}\n\n"
            f"当前得分：{scores.get(current_sector, float('nan')):.4f}\n"
            f"最强板块：{top_sector}（{top_score:.4f}）\n"
            f"未达切换阈值（需 >{scores.get(current_sector, 0)*THRESHOLD:.4f}）\n\n"
            f"趋势过滤：{'做多' if in_market else '现金'}\n"
            f"─── 无需操作 ───"
        )

    elif action == "BUY":
        title = f"[量化信号] {date_str} — 建仓 {new_sector}"
        content = (
            f"月末信号 · {date_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"操作：建仓 (BUY)\n\n"
            f"买入板块：{new_sector}\n"
            f"对应ETF ：{new_etf}\n"
            f"综合得分：{top_score:.4f}\n\n"
            f"趋势过滤：做多\n"
            f"─── 请执行买入 ───"
        )

    elif action == "SWITCH":
        title = f"[量化信号] {date_str} — 换仓 {current_sector}→{new_sector}"
        content = (
            f"月末信号 · {date_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"操作：换仓 (SWITCH)\n\n"
            f"卖出：{current_sector} / ETF {current_etf}\n"
            f"买入：{new_sector}  / ETF {new_etf}\n\n"
            f"新板块得分：{top_score:.4f}\n"
            f"旧板块得分：{scores.get(current_sector, float('nan')):.4f}\n"
            f"阈值倍数：×{THRESHOLD}（已超越）\n\n"
            f"趋势过滤：做多\n"
            f"─── 请先卖出再买入 ───"
        )

    elif action in ("SELL_ALL", "CASH"):
        if action == "SELL_ALL":
            title = f"[量化信号] {date_str} — 清仓转现金"
            content = (
                f"月末信号 · {date_str}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"操作：清仓 (SELL ALL)\n\n"
                f"卖出：{current_sector} / ETF {current_etf}\n\n"
                f"原因：趋势过滤触发（CSI300 跌破{TREND_WIN}月均线）\n"
                f"─── 全部换成货币基金/现金 ───"
            )
        else:
            title = f"[量化信号] {date_str} — 继续持有现金"
            content = (
                f"月末信号 · {date_str}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"操作：持有现金 (CASH)\n\n"
                f"趋势过滤触发，市场仍处于均线以下。\n"
                f"─── 无需操作 ───"
            )
    else:
        title   = f"[量化信号] {date_str} — {action}"
        content = f"未知动作: {action}"

    print(f"\n{'─'*40}")
    print(f"Title  : {title}")
    print(f"Content:\n{content}")
    print(f"{'─'*40}\n")

    # ── 8. Push to WeChat ─────────────────────────────────────────────────────
    push_wechat(title, content)

    # ── 9. Persist new holdings ───────────────────────────────────────────────
    save_holdings(new_sector, new_etf, date_str)

    print("Done.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monthly sector rotation signal + WeChat push")
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass last-trading-day check and always run"
    )
    args = parser.parse_args()
    run(force=args.force)
