from __future__ import annotations

import datetime
import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from config import MAX_DTE, MIN_DTE, OTM_RANGE_PCT

logger = logging.getLogger(__name__)


def _load_app_module():
    import app as webapp

    return webapp


def _safe_ticker_info(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception as exc:
        logger.debug("%s info fetch failed: %s", symbol, exc)
        return {}


def _fallback_history(symbol: str, count: int = 120) -> pd.DataFrame:
    webapp = _load_app_module()
    hist, source = webapp._daily_history(symbol, count)
    if hist is None or hist.empty:
        logger.warning("%s fallback history unavailable", symbol)
        return pd.DataFrame()
    logger.info("%s using fallback history source: %s", symbol, source)
    return hist


def _chunked(items: list[str], size: int):
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _longbridge_quotes_with_retry(symbols: list[str]) -> dict[str, object]:
    if not symbols:
        return {}

    webapp = _load_app_module()
    ctx = webapp._lb_ctx()
    if ctx is None:
        return {}

    out: dict[str, object] = {}
    for chunk in _chunked(symbols, 20):
        for attempt in range(2):
            try:
                quotes = ctx.option_quote(chunk)
                for quote in quotes:
                    key = getattr(quote, "symbol", "")
                    if key:
                        out[key] = quote
                break
            except Exception as exc:
                is_rate_limit = "301607" in str(exc)
                if attempt == 0 and is_rate_limit:
                    logger.warning("Longbridge option quotes rate-limited, waiting 65s before retry")
                    time.sleep(65)
                    continue
                logger.warning("Longbridge option_quote failed for chunk(size=%s): %s", len(chunk), exc)
                break
        time.sleep(0.3)
    return out


def get_stock_info(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="90d")
    if hist.empty:
        hist = _fallback_history(symbol, 90)
    if hist.empty:
        raise ValueError(f"无法获取 {symbol} 历史数据")

    price = hist["Close"].iloc[-1]
    daily_returns = hist["Close"].pct_change().dropna()
    hv30 = daily_returns.tail(30).std() * (252 ** 0.5) * 100
    week52_high = hist["High"].max()
    week52_low = hist["Low"].min()

    info = _safe_ticker_info(symbol)
    market_cap = info.get("marketCap")

    return {
        "symbol": symbol,
        "price": round(float(price), 2),
        "hv30": round(float(hv30), 2),
        "week52_high": round(float(week52_high), 2),
        "week52_low": round(float(week52_low), 2),
        "market_cap": market_cap,
    }


def get_company_profile(symbol: str) -> dict:
    info = _safe_ticker_info(symbol)
    return {
        "symbol": symbol,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "long_name": info.get("longName") or info.get("shortName") or symbol,
        "market_cap": info.get("marketCap"),
        "average_volume": info.get("averageVolume"),
        "beta": info.get("beta"),
    }


def get_recent_news(symbol: str, limit: int = 5) -> list[dict]:
    ticker = yf.Ticker(symbol)
    try:
        raw = ticker.news or []
    except Exception as exc:
        logger.debug("%s news fetch failed: %s", symbol, exc)
        return []

    news: list[dict] = []
    for item in raw[:limit]:
        title = item.get("title") or item.get("content", {}).get("title") or ""
        summary = item.get("summary") or item.get("content", {}).get("summary") or ""
        publisher = item.get("publisher") or item.get("content", {}).get("publisher") or ""
        link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url") or ""
        provider_time = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
        news.append(
            {
                "title": title,
                "summary": summary,
                "publisher": publisher,
                "link": link,
                "published_at": provider_time,
            }
        )
    return news


def get_price_history(symbol: str, period: str = "90d", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty and interval == "1d":
        try:
            days = int(str(period).rstrip("d"))
        except Exception:
            days = 90
        hist = _fallback_history(symbol, max(days, 30))
    return hist


def _yahoo_option_chain(symbol: str, spot: float) -> Optional[pd.DataFrame]:
    ticker = yf.Ticker(symbol)
    today = datetime.date.today()

    try:
        all_expiries = ticker.options
    except Exception as exc:
        logger.warning("%s yahoo options unavailable: %s", symbol, exc)
        return None

    if not all_expiries:
        return None

    frames = []
    for exp_str in all_expiries:
        exp_date = datetime.date.fromisoformat(exp_str)
        dte = (exp_date - today).days
        if dte < MIN_DTE or dte > MAX_DTE:
            continue

        try:
            chain = ticker.option_chain(exp_str)
        except Exception as exc:
            logger.warning("%s %s yahoo option fetch failed: %s", symbol, exp_str, exc)
            continue

        for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
            if df.empty:
                continue
            item = df.copy()
            item["optionType"] = opt_type
            item["expiry"] = exp_str
            item["dte"] = dte
            item["spotPrice"] = spot
            frames.append(item)

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    lo = spot * (1 - OTM_RANGE_PCT)
    hi = spot * (1 + OTM_RANGE_PCT)
    result = result[(result["strike"] >= lo) & (result["strike"] <= hi)].copy()
    if result.empty:
        return None

    result["moneyness"] = (result["strike"] - result["spotPrice"]) / result["spotPrice"]
    result["moneyness_pct"] = result["moneyness"].map(lambda x: f"{x:+.1%}")
    result.rename(
        columns={
            "impliedVolatility": "iv",
            "openInterest": "oi",
            "lastPrice": "lastPrice",
        },
        inplace=True,
    )
    if "iv" in result.columns:
        result["iv_pct"] = (result["iv"] * 100).round(2)
    result["symbol"] = symbol
    return result


def _longbridge_option_chain(symbol: str, spot: float) -> Optional[pd.DataFrame]:
    webapp = _load_app_module()
    base = webapp._longbridge_option_candidates(
        symbol,
        spot,
        MIN_DTE,
        MAX_DTE,
        "both",
        OTM_RANGE_PCT,
        enrich_quotes=False,
    )
    if base is None or base.empty:
        return None

    base = base.copy()
    if "lb_symbol" in base.columns:
        lb_symbols = [s for s in base["lb_symbol"].dropna().unique().tolist() if s]
    else:
        lb_symbols = []

    quotes = _longbridge_quotes_with_retry(lb_symbols)
    if quotes:
        quote_rows = []
        for lb_symbol, quote in quotes.items():
            item = webapp._extract_option_mark_from_quote(quote)
            item["lb_symbol"] = lb_symbol
            quote_rows.append(item)
        quote_df = pd.DataFrame(quote_rows)
        if not quote_df.empty:
            drop_cols = [c for c in ("bid", "ask", "last", "volume", "oi", "iv_pct") if c in base.columns]
            base = base.drop(columns=drop_cols).merge(quote_df, on="lb_symbol", how="left")

    base["spotPrice"] = spot
    base["moneyness"] = (pd.to_numeric(base["strike"], errors="coerce") - spot) / spot
    base["moneyness_pct"] = base["moneyness"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
    base["iv_pct"] = pd.to_numeric(base.get("iv_pct"), errors="coerce")
    base["iv"] = base["iv_pct"] / 100.0
    base["lastPrice"] = pd.to_numeric(base.get("last"), errors="coerce")
    base["symbol"] = symbol
    return base


def get_option_chain(symbol: str) -> Optional[pd.DataFrame]:
    info = get_stock_info(symbol)
    spot = info["price"]

    yahoo_result = _yahoo_option_chain(symbol, spot)
    if yahoo_result is not None and not yahoo_result.empty:
        return yahoo_result

    result = _longbridge_option_chain(symbol, spot)
    if result is None or result.empty:
        logger.warning("%s: 符合条件的期权链为空", symbol)
        return None
    return result
