# ============================================================
#  iv_analyzer.py  —  隐含波动率 (IV) 分析
# ============================================================
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from config import IV_HIGH_PERCENTILE, IV_LOW_PERCENTILE

logger = logging.getLogger(__name__)


def analyze_iv(chain_df: pd.DataFrame) -> dict:
    """
    对期权链做 IV 统计分析，返回汇总字典
    包括：IV 均值、中位数、高/低百分位、IV Rank、IV Skew、建议信号
    """
    if chain_df is None or chain_df.empty or "iv_pct" not in chain_df.columns:
        return {}

    iv_series = chain_df["iv_pct"].dropna()
    if iv_series.empty:
        return {}

    iv_mean   = round(iv_series.mean(), 2)
    iv_median = round(iv_series.median(), 2)
    iv_std    = round(iv_series.std(), 2)
    iv_max    = round(iv_series.max(), 2)
    iv_min    = round(iv_series.min(), 2)

    p25 = round(np.percentile(iv_series, 25), 2)
    p75 = round(np.percentile(iv_series, 75), 2)

    # IV Rank：当前 IV 在近期分布中的排名（0-100）
    iv_rank = round((iv_mean - iv_min) / (iv_max - iv_min) * 100, 1) if iv_max != iv_min else 50.0

    # IV Skew：Put IV 均值 - Call IV 均值（正值=看跌偏斜）
    iv_skew = None
    if "optionType" in chain_df.columns:
        call_iv = chain_df[chain_df["optionType"] == "call"]["iv_pct"].mean()
        put_iv  = chain_df[chain_df["optionType"] == "put"]["iv_pct"].mean()
        if pd.notna(call_iv) and pd.notna(put_iv):
            iv_skew = round(put_iv - call_iv, 2)

    # 信号判断
    signal = _iv_signal(iv_rank, iv_skew)

    # ATM IV（行权价最接近现价）
    if "spotPrice" in chain_df.columns and "strike" in chain_df.columns:
        spot = chain_df["spotPrice"].iloc[0]
        atm_df = chain_df.copy()
        atm_df["dist"] = (atm_df["strike"] - spot).abs()
        atm_row = atm_df.loc[atm_df["dist"].idxmin()]
        atm_iv = round(atm_row["iv_pct"], 2) if pd.notna(atm_row["iv_pct"]) else None
    else:
        atm_iv = None

    return {
        "iv_mean":   iv_mean,
        "iv_median": iv_median,
        "iv_std":    iv_std,
        "iv_max":    iv_max,
        "iv_min":    iv_min,
        "iv_p25":    p25,
        "iv_p75":    p75,
        "iv_rank":   iv_rank,
        "atm_iv":    atm_iv,
        "iv_skew":   iv_skew,
        "signal":    signal,
    }


def _iv_signal(iv_rank: float, iv_skew) -> str:
    """根据 IV Rank 和 Skew 生成交易倾向信号"""
    lines = []

    if iv_rank >= IV_HIGH_PERCENTILE:
        lines.append("📈 IV偏高 → 适合做空波动率（卖权/Iron Condor）")
    elif iv_rank <= IV_LOW_PERCENTILE:
        lines.append("📉 IV偏低 → 适合做多波动率（买权/Straddle）")
    else:
        lines.append("➡️ IV中性 → 趋势策略为主（垂直价差）")

    if iv_skew is not None:
        if iv_skew > 3:
            lines.append("⚠️ Put偏斜显著 → 市场对下行保护需求高")
        elif iv_skew < -3:
            lines.append("🚀 Call偏斜显著 → 市场对上行期望高")

    return " | ".join(lines) if lines else "无明显信号"


def top_iv_contracts(chain_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """返回 IV 最高的 N 个合约（值得关注的热门合约）"""
    if chain_df is None or chain_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["symbol","optionType","expiry","dte","strike","moneyness_pct","iv_pct","oi","volume","lastPrice"] if c in chain_df.columns]
    return chain_df.nlargest(n, "iv_pct")[cols].reset_index(drop=True)


def high_oi_contracts(chain_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """返回 OI 最高的 N 个合约（主力持仓）"""
    if chain_df is None or chain_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["symbol","optionType","expiry","dte","strike","moneyness_pct","iv_pct","oi","volume","lastPrice"] if c in chain_df.columns]
    return chain_df.nlargest(n, "oi")[cols].reset_index(drop=True)
