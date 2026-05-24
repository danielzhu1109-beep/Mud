# ============================================================
#  greeks_calculator.py  —  Black-Scholes Greeks 计算
# ============================================================
from __future__ import annotations

import math
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import RISK_FREE_RATE

logger = logging.getLogger(__name__)


# ── Black-Scholes 核心函数 ──────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """计算 d1、d2"""
    if T <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call") -> float:
    """Black-Scholes 理论价格"""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return 0.0
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """
    计算完整 Greeks
    返回：delta, gamma, theta(每日), vega(每1%IV变动), rho
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}

    nd1  = norm.cdf(d1)
    nd2  = norm.cdf(d2)
    npd1 = norm.pdf(d1)
    nd1n = norm.cdf(-d1)
    nd2n = norm.cdf(-d2)

    gamma = npd1 / (S * sigma * math.sqrt(T))
    vega  = S * npd1 * math.sqrt(T) / 100   # 每1%IV变动的价格变化

    if option_type == "call":
        delta = nd1
        theta = (-(S * npd1 * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * nd2) / 365
        rho   = K * T * math.exp(-r * T) * nd2 / 100
    else:
        delta = nd1 - 1
        theta = (-(S * npd1 * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * nd2n) / 365
        rho   = -K * T * math.exp(-r * T) * nd2n / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "rho":   round(rho, 4),
    }


# ── 批量计算 DataFrame ──────────────────────────────────────

def calculate_greeks_df(chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    对整个期权链 DataFrame 批量计算 Greeks
    需要列：spotPrice, strike, dte, iv, optionType
    """
    if chain_df is None or chain_df.empty:
        return chain_df

    results = []
    for _, row in chain_df.iterrows():
        try:
            S     = float(row["spotPrice"])
            K     = float(row["strike"])
            T     = float(row["dte"]) / 365.0
            sigma = float(row.get("iv", 0)) if pd.notna(row.get("iv")) else 0
            r     = RISK_FREE_RATE
            opt   = str(row.get("optionType", "call"))

            if T <= 0 or sigma <= 0:
                g = {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
            else:
                g = bs_greeks(S, K, T, r, sigma, opt)

            # 理论价格 vs 市场价
            theory_price = round(bs_price(S, K, T, r, sigma, opt), 4) if T > 0 and sigma > 0 else None
            market_price = row.get("lastPrice", None)
            premium_diff = None
            if theory_price and market_price and pd.notna(market_price):
                premium_diff = round(float(market_price) - theory_price, 4)

        except Exception as e:
            logger.debug(f"Greeks 计算异常: {e}")
            g = {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
            theory_price = None
            premium_diff = None

        results.append({**g, "theory_price": theory_price, "premium_diff": premium_diff})

    greeks_df = pd.DataFrame(results)
    return pd.concat([chain_df.reset_index(drop=True), greeks_df], axis=1)


def summarize_greeks(chain_df: pd.DataFrame) -> dict:
    """汇总期权链整体 Greeks 特征"""
    if chain_df is None or chain_df.empty:
        return {}

    def safe_mean(col):
        if col in chain_df.columns:
            return round(chain_df[col].dropna().mean(), 4)
        return None

    # 按 Call/Put 分开统计 Delta
    call_delta = put_delta = None
    if "optionType" in chain_df.columns and "delta" in chain_df.columns:
        call_delta = round(chain_df[chain_df["optionType"] == "call"]["delta"].dropna().mean(), 4)
        put_delta  = round(chain_df[chain_df["optionType"] == "put"]["delta"].dropna().mean(), 4)

    return {
        "avg_delta_call": call_delta,
        "avg_delta_put":  put_delta,
        "avg_gamma":      safe_mean("gamma"),
        "avg_theta":      safe_mean("theta"),
        "avg_vega":       safe_mean("vega"),
    }
