from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import threading
import uuid
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from scipy.stats import norm
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from data_fetcher import get_company_profile, get_recent_news

try:
    from longbridge.openapi import Config as LBConfig, Market, QuoteContext, Period, AdjustType
except Exception:  # pragma: no cover
    LBConfig = None
    Market = None
    QuoteContext = None
    Period = None
    AdjustType = None

try:
    from webull.core.client import ApiClient as WebullApiClient
    from webull.trade.trade_client import TradeClient as WebullTradeClient
    from webull.data.data_client import DataClient as WebullDataClient
    from webull.data.common.category import Category as WebullCategory
    from webull.data.common.timespan import Timespan as WebullTimespan
except Exception:  # pragma: no cover
    WebullApiClient = None
    WebullTradeClient = None
    WebullDataClient = None
    WebullCategory = None
    WebullTimespan = None


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.errorhandler(Exception)
def _api_json_error(exc):
    if request.path.startswith("/api/"):
        logger.exception("api error on %s", request.path)
        return jsonify({"error": str(exc), "path": request.path}), 500
    if isinstance(exc, HTTPException):
        return exc
    raise exc

ET = pytz.timezone("America/New_York")
MIN_DTE = int(os.getenv("MIN_DTE", "7"))
MAX_DTE = int(os.getenv("MAX_DTE", "60"))
OTM_RANGE_PCT = float(os.getenv("OTM_RANGE_PCT", "0.15"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
CHART_LIBRARY_DIR = Path(os.getenv("CHART_LIBRARY_DIR", "chart_library"))
CHART_LIBRARY_INBOX = CHART_LIBRARY_DIR / "incoming"
CHART_LIBRARY_ARCHIVE = CHART_LIBRARY_DIR / "archive"
CHART_LIBRARY_MANIFEST = CHART_LIBRARY_DIR / "manifest.json"
CHART_LIBRARY_PROFILE_CACHE = CHART_LIBRARY_DIR / "profile_cache.json"
MARKETCAP_CACHE = Path(os.getenv("MARKETCAP_CACHE", "cache/marketcap_top1000.json"))
TOP50_CACHE = Path(os.getenv("TOP50_CACHE", "cache/universe_top50.json"))
UNUSUAL_OPTIONS_CACHE = Path(os.getenv("UNUSUAL_OPTIONS_CACHE", "cache/unusual_options_daily.json"))
MARKET_ENV_CACHE = Path(os.getenv("MARKET_ENV_CACHE", "cache/market_environment.json"))
MARKETCAP_SOURCE = "https://companiesmarketcap.com/usd/"
MARKETCAP_MAX_PAGES = int(os.getenv("MARKETCAP_MAX_PAGES", "10"))
MARKETCAP_PAGE_SIZE = int(os.getenv("MARKETCAP_PAGE_SIZE", "100"))
UNIVERSE_SCAN_LIMIT = int(os.getenv("UNIVERSE_SCAN_LIMIT", "1000"))
UNUSUAL_SCAN_SYMBOL_LIMIT = int(os.getenv("UNUSUAL_SCAN_SYMBOL_LIMIT", "90"))
UNUSUAL_PREFILTER_MULTIPLIER = float(os.getenv("UNUSUAL_PREFILTER_MULTIPLIER", "2.5"))
UNUSUAL_MAX_TOP_PER_SYMBOL = int(os.getenv("UNUSUAL_MAX_TOP_PER_SYMBOL", "8"))
UNUSUAL_MIN_TARGET_ROWS = int(os.getenv("UNUSUAL_MIN_TARGET_ROWS", "30"))
LONGBRIDGE_OPTION_QUOTE_CHUNK = int(os.getenv("LONGBRIDGE_OPTION_QUOTE_CHUNK", "80"))
SIM_TRADING_DIR = Path(os.getenv("SIM_TRADING_DIR", "sim_trading"))
SIM_TRADING_STATE = SIM_TRADING_DIR / "sim_state.json"
LEARNING_STATE = SIM_TRADING_DIR / "learning_state.json"
SIGNAL_MEMORY = SIM_TRADING_DIR / "signal_memory.json"
KNOWLEDGE_WIKI_DIR = Path(os.getenv("KNOWLEDGE_WIKI_DIR", "knowledge_wiki"))
KNOWLEDGE_WIKI_INDEX = KNOWLEDGE_WIKI_DIR / "README.md"
KNOWLEDGE_WIKI_FACTORS = KNOWLEDGE_WIKI_DIR / "factors.md"
KNOWLEDGE_WIKI_SOURCES = KNOWLEDGE_WIKI_DIR / "sources.md"
KNOWLEDGE_WIKI_SIGNALS = KNOWLEDGE_WIKI_DIR / "signals.md"
KNOWLEDGE_WIKI_REPORTS = KNOWLEDGE_WIKI_DIR / "reports.md"
KNOWLEDGE_WIKI_LEARNING = KNOWLEDGE_WIKI_DIR / "learning.md"
KNOWLEDGE_WIKI_MARKET = KNOWLEDGE_WIKI_DIR / "market_memory.md"
SIM_TRADE_MULTIPLIER = int(os.getenv("SIM_TRADE_MULTIPLIER", "100"))
WEBULL_LIVE_CONFIG = Path(os.getenv("WEBULL_LIVE_CONFIG", "webull_live_config.json"))
WEBULL_LIVE_STATE = Path(os.getenv("WEBULL_LIVE_STATE", "webull_live_state.json"))
SIM_TRADING_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_WIKI_DIR.mkdir(parents=True, exist_ok=True)
WEBULL_LIVE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
WEBULL_LIVE_STATE.parent.mkdir(parents=True, exist_ok=True)
SIM_STATE_LOCK = threading.Lock()

for _path in (CHART_LIBRARY_DIR, CHART_LIBRARY_INBOX, CHART_LIBRARY_ARCHIVE, MARKETCAP_CACHE.parent, UNUSUAL_OPTIONS_CACHE.parent, KNOWLEDGE_WIKI_DIR, MARKET_ENV_CACHE.parent):
    _path.mkdir(parents=True, exist_ok=True)


def _safe(value: Any, digits: int = 2):
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _short_money(value: Any) -> str:
    amount = _num(value, 0.0)
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:.0f}"


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "SPY").strip().upper()


def _symbol_root(symbol: str) -> str:
    s = _normalize_symbol(symbol)
    if s.endswith(".US"):
        s = s[:-3]
    return s


def _to_yfinance_symbol(symbol: str) -> str:
    return _symbol_root(symbol).replace(".", "-")


def _lb_underlying_symbol(symbol: str) -> str:
    return f"{_symbol_root(symbol)}.US"


def _lb_option_symbol(symbol: str, expiry: str, strike: float, opt_type: str) -> str:
    root = _symbol_root(symbol)
    yymmdd = dt.date.fromisoformat(expiry).strftime("%y%m%d")
    side = "C" if opt_type.lower() == "call" else "P"
    strike_code = f"{int(round(float(strike) * 1000)):06d}"
    return f"{root}{yymmdd}{side}{strike_code}.US"


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    try:
        if getattr(out.index, "tz", None) is not None:
            out.index = out.index.tz_localize(None)
    except Exception:
        pass
    return out


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _parse_utc_datetime(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        parsed = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


@lru_cache(maxsize=1)
def _lb_ctx() -> Optional[QuoteContext]:
    if LBConfig is None or QuoteContext is None:
        return None

    app_key = os.getenv("LONGBRIDGE_APP_KEY")
    app_secret = os.getenv("LONGBRIDGE_APP_SECRET")
    access_token = os.getenv("LONGBRIDGE_ACCESS_TOKEN")
    if not all([app_key, app_secret, access_token]):
        return None

    try:
        cfg = LBConfig.from_apikey(
            app_key,
            app_secret,
            access_token,
            enable_print_quote_packages=False,
        )
        return QuoteContext(cfg)
    except Exception as exc:
        logger.warning("Longbridge init failed: %s", exc)
        return None


def _lb_market_enum(market: str):
    if Market is None:
        return None
    mapping = {
        "US": Market.US,
        "HK": Market.HK,
        "CN": Market.CN,
        "SG": Market.SG,
    }
    return mapping.get(market.upper(), Market.US)


def _yf_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(_to_yfinance_symbol(symbol))


def _yahoo_history(symbol: str, period: str, interval: str | None = None) -> pd.DataFrame:
    ticker = _yf_ticker(symbol)
    kwargs = {"period": period}
    if interval:
        kwargs["interval"] = interval
    hist = ticker.history(**kwargs)
    return _normalize_index(hist)


def _fetch_json(url: str, params: dict[str, Any], timeout: int = 25) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("request failed %s: %s", url, exc)
        return None


def _twelvedata_history(symbol: str, interval: str = "1day", outputsize: int = 120, api_key: str | None = None) -> pd.DataFrame:
    key = (api_key or TWELVEDATA_API_KEY).strip()
    if not key:
        return _empty_df()
    payload = _fetch_json(
        "https://api.twelvedata.com/time_series",
        {
            "symbol": _normalize_symbol(symbol).split(".")[0],
            "interval": interval,
            "outputsize": outputsize,
            "apikey": key,
            "format": "JSON",
        },
    )
    if not payload or payload.get("status") == "error" or "values" not in payload:
        return _empty_df()
    rows = []
    for item in payload.get("values", []):
        ts = pd.to_datetime(item.get("datetime"), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        rows.append(
            {
                "t": ts.tz_convert(None),
                "Open": float(item.get("open", np.nan)),
                "High": float(item.get("high", np.nan)),
                "Low": float(item.get("low", np.nan)),
                "Close": float(item.get("close", np.nan)),
                "Volume": float(item.get("volume", np.nan)),
            }
        )
    if not rows:
        return _empty_df()
    return pd.DataFrame(rows).set_index("t").sort_index()


def _alpha_vantage_history(symbol: str, function: str = "TIME_SERIES_DAILY_ADJUSTED", interval: str = "5min", api_key: str | None = None) -> pd.DataFrame:
    key = (api_key or ALPHAVANTAGE_API_KEY).strip()
    if not key:
        return _empty_df()

    params = {
        "function": function,
        "symbol": _normalize_symbol(symbol).split(".")[0],
        "apikey": key,
        "outputsize": "compact",
    }
    if function == "TIME_SERIES_INTRADAY":
        params["interval"] = interval

    payload = _fetch_json("https://www.alphavantage.co/query", params)
    if not payload:
        return _empty_df()

    series = None
    for k, v in payload.items():
        if "Time Series" in k:
            series = v
            break
    if not series:
        return _empty_df()

    rows = []
    for ts_str, item in series.items():
        ts = pd.to_datetime(ts_str, errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        rows.append(
            {
                "t": ts.tz_convert(None),
                "Open": float(item.get("1. open", np.nan)),
                "High": float(item.get("2. high", np.nan)),
                "Low": float(item.get("3. low", np.nan)),
                "Close": float(item.get("4. close", np.nan)),
                "Volume": float(item.get("6. volume", item.get("5. volume", np.nan))),
            }
        )
    if not rows:
        return _empty_df()
    return pd.DataFrame(rows).set_index("t").sort_index()


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _infer_library_tags(text: str, filename: str = "") -> list[str]:
    content = f"{filename}\n{text}".lower()
    tags: list[str] = []

    def add(tag: str):
        if tag not in tags:
            tags.append(tag)

    if any(k in content for k in ("bullish", "call", "calls", "long", "breakout", "break out", "buy call")):
        add("bullish")
    if any(k in content for k in ("bearish", "put", "puts", "short", "breakdown", "sell put")):
        add("bearish")
    if any(k in content for k in ("breakout", "new high", "resistance break", "flag")):
        add("breakout")
    if any(k in content for k in ("breakdown", "new low", "support break")):
        add("breakdown")
    if any(k in content for k in ("7dte", "0dte", "weekly", "this week")):
        add("7dte")
    if any(k in content for k in ("30dte", "monthly", "next month")):
        add("30dte")
    if any(k in content for k in ("high iv", "rich iv", "iv rank 70", "ivr 70", "volatility high")):
        add("high_iv")
    if any(k in content for k in ("low iv", "cheap iv", "iv rank 30", "ivr 30", "volatility low")):
        add("low_iv")
    if any(k in content for k in ("win", "profit", "take profit", "target hit", "success")):
        add("success")
    if any(k in content for k in ("loss", "stop loss", "failed", "fail")):
        add("fail")
    if "call" in content and "put" not in content:
        add("call")
    if "put" in content and "call" not in content:
        add("put")
    return tags


def _load_chart_manifest() -> dict[str, Any]:
    if CHART_LIBRARY_MANIFEST.exists():
        try:
            return json.loads(CHART_LIBRARY_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_chart_manifest(manifest: dict[str, Any]) -> None:
    CHART_LIBRARY_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_chart_profile_cache() -> dict[str, Any]:
    if CHART_LIBRARY_PROFILE_CACHE.exists():
        try:
            payload = json.loads(CHART_LIBRARY_PROFILE_CACHE.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _save_chart_profile_cache(payload: dict[str, Any]) -> None:
    try:
        CHART_LIBRARY_PROFILE_CACHE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("chart profile cache save failed: %s", exc)


def _chart_library_files() -> list[dict[str, Any]]:
    manifest = _load_chart_manifest()
    files = []
    for folder in (CHART_LIBRARY_INBOX, CHART_LIBRARY_ARCHIVE):
        for path in sorted(folder.glob("*")):
            if not path.is_file() or not _is_image_file(path):
                continue
            meta = manifest.get(path.name, {})
            files.append(
                {
                    "name": path.name,
                    "folder": folder.name,
                    "url": f"/api/library/file/{path.name}",
                    "size": path.stat().st_size,
                    "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "tags": meta.get("tags", []),
                    "notes": meta.get("notes", ""),
                    "symbol": meta.get("symbol", ""),
                    "status": meta.get("status", "unlabeled"),
                    "ocr_text": meta.get("ocr_text", ""),
                    "ocr_summary": meta.get("ocr_summary", ""),
                    "auto_tags": meta.get("auto_tags", []),
                    "source_folder": meta.get("source_folder", ""),
                    "import_mode": meta.get("import_mode", "manual"),
                }
            )
    return files


def _update_chart_manifest(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_chart_manifest()
    entry = manifest.get(filename, {})
    entry.update(payload)
    manifest[filename] = entry
    _save_chart_manifest(manifest)
    return entry


def _chart_library_signature(files: list[dict[str, Any]], include_ai_summary: bool = False) -> str:
    payload = [
        {
            "name": item.get("name"),
            "folder": item.get("folder"),
            "mtime": item.get("mtime"),
            "tags": item.get("tags", []),
            "notes": item.get("notes", ""),
            "status": item.get("status", ""),
            "ocr_summary": item.get("ocr_summary", ""),
            "auto_tags": item.get("auto_tags", []),
            "include_ai_summary": include_ai_summary,
        }
        for item in files
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _chart_library_profile(force_refresh: bool = False, include_ai_summary: bool = False) -> dict[str, Any]:
    files = _chart_library_files()
    signature = _chart_library_signature(files, include_ai_summary=include_ai_summary)
    if not force_refresh:
        cached = _load_chart_profile_cache()
        if cached.get("signature") == signature and isinstance(cached.get("profile"), dict):
            return cached["profile"]

    tag_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    ocr_fragments: list[str] = []
    for item in files:
        status_counts[item.get("status", "unlabeled")] = status_counts.get(item.get("status", "unlabeled"), 0) + 1
        for tag in item.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        for tag in item.get("auto_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if item.get("ocr_summary"):
            ocr_fragments.append(f"{item['name']}: {item.get('ocr_summary')}")
        elif item.get("ocr_text"):
            ocr_fragments.append(f"{item['name']}: {item.get('ocr_text')[:500]}")

    summary = {
        "total": len(files),
        "labeled": sum(1 for item in files if item.get("status") != "unlabeled"),
        "status_counts": status_counts,
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "focus_tags": [tag for tag, count in sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]],
    }

    notes = [
        f"{item['name']}: {','.join(item.get('tags', []))} | {item.get('notes', '')}"
        for item in files
        if item.get("notes") or item.get("tags")
    ]
    if ocr_fragments:
        notes.extend(ocr_fragments[:50])
    if notes and include_ai_summary:
        ai_text = _deepseek_chat(
            "下面是我历史K线图/期权截图经验库的标签、备注和OCR摘要。"
            "请你像一个交易风格分析器一样，提炼出我的："
            "1) 选股/选期权偏好；2) 常见入场方式；3) 常见止损方式；4) 常见止盈方式；"
            "5) 偏好的 DTE / IV / 方向 / 结构；6) 风险厌恶程度；7) 最适合我的筛选规则。"
            "要求输出中文，尽量具体，最好能直接用于后面的期权排名与打分。\n"
            + "\n".join(notes[:60]),
        )
        if ai_text:
            summary["ai_summary"] = ai_text
    summary["conclusion"] = _chart_library_conclusion(summary)
    _save_chart_profile_cache(
        {
            "signature": signature,
            "updated_at": _now_et_iso(),
            "profile": summary,
        }
    )
    return summary


def _chart_library_bias(profile: dict[str, Any]) -> dict[str, float]:
    tags = profile.get("tag_counts", {})
    bullish = tags.get("bullish", 0) + tags.get("breakout", 0) + tags.get("call", 0)
    bearish = tags.get("bearish", 0) + tags.get("breakdown", 0) + tags.get("put", 0)
    iv_high = tags.get("high_iv", 0)
    iv_low = tags.get("low_iv", 0)
    dte_short = tags.get("7dte", 0) + tags.get("weekly", 0)
    dte_mid = tags.get("30dte", 0) + tags.get("monthly", 0)
    return {
        "direction_bias": float(bullish - bearish),
        "iv_bias": float(iv_high - iv_low),
        "dte_bias": float(dte_short - dte_mid),
    }


def _chart_library_conclusion(profile: dict[str, Any]) -> dict[str, Any]:
    tags = profile.get("tag_counts", {})
    total = int(profile.get("total", 0) or 0)
    labeled = int(profile.get("labeled", 0) or 0)

    bullish = tags.get("bullish", 0) + tags.get("breakout", 0) + tags.get("call", 0)
    bearish = tags.get("bearish", 0) + tags.get("breakdown", 0) + tags.get("put", 0)
    iv_high = tags.get("high_iv", 0)
    iv_low = tags.get("low_iv", 0)
    dte_short = tags.get("7dte", 0) + tags.get("weekly", 0)
    dte_mid = tags.get("30dte", 0) + tags.get("monthly", 0)

    if bullish > bearish * 1.2:
        direction = "bullish"
    elif bearish > bullish * 1.2:
        direction = "bearish"
    else:
        direction = "neutral"

    if dte_short > dte_mid * 1.2 and dte_short > 0:
        dte_style = "short"
    elif dte_mid > dte_short * 1.2 and dte_mid > 0:
        dte_style = "mid"
    else:
        dte_style = "mixed"

    if iv_high > iv_low * 1.2:
        iv_style = "high"
    elif iv_low > iv_high * 1.2:
        iv_style = "low"
    else:
        iv_style = "balanced"

    structure = "call" if direction == "bullish" else "put" if direction == "bearish" else "both"
    if bullish >= bearish * 1.4:
        entry_style = "breakout_or_pullback_long"
        risk_style = "tight_stop"
    elif bearish >= bullish * 1.4:
        entry_style = "breakdown_or_rally_short"
        risk_style = "fast_exit"
    else:
        entry_style = "range_or_confirmation"
        risk_style = "balanced"

    if iv_high > iv_low:
        structure_hint = "prefer_credit_or_debit_spread" if direction == "neutral" else "prefer_directional_with_liquidity"
    else:
        structure_hint = "prefer_directional_or_limited_risk"

    confidence = min(0.95, 0.2 + labeled / 20.0)
    summary = profile.get("ai_summary") or ""
    if not summary:
        summary = (
            f"你的经验库更偏向{ '看涨' if direction == 'bullish' else '看跌' if direction == 'bearish' else '双向' }，"
            f"DTE 偏好为{dte_style}，IV 偏好为{iv_style}，入场风格偏向{entry_style}。"
        )

    return {
        "direction": direction,
        "dte_style": dte_style,
        "iv_style": iv_style,
        "structure": structure,
        "entry_style": entry_style,
        "risk_style": risk_style,
        "structure_hint": structure_hint,
        "confidence": round(confidence, 2),
        "summary": summary,
        "signals": {
            "bullish": bullish,
            "bearish": bearish,
            "iv_high": iv_high,
            "iv_low": iv_low,
            "dte_short": dte_short,
            "dte_mid": dte_mid,
        },
        "scope": "stock_and_options",
        "sample_size": total,
    }


def _fetch_marketcap_page(page: int) -> list[dict[str, Any]]:
    url = MARKETCAP_SOURCE if page == 1 else f"{MARKETCAP_SOURCE}page/{page}/"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=35)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for tr in soup.select("table tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 8:
            continue
        rank_cell = cells[1].get("data-sort") or cells[1].get_text(" ", strip=True)
        try:
            rank = int(re.sub(r"\D", "", str(rank_cell)))
        except Exception:
            continue
        name_div = cells[2]
        company_name = name_div.select_one(".company-name")
        company_code = name_div.select_one(".company-code")
        if not company_name or not company_code:
            continue
        name = company_name.get_text(" ", strip=True)
        ticker = company_code.get_text(" ", strip=True)
        market_cap_raw = cells[3].get("data-sort")
        price_raw = cells[4].get("data-sort")
        change_raw = cells[5].get("data-sort")
        country = cells[7].get_text(" ", strip=True)
        rows.append(
            {
                "rank": rank,
                "name": name,
                "symbol": ticker.strip().replace(" ", ""),
                "market_cap": float(market_cap_raw) if market_cap_raw else None,
                "price": float(price_raw) / 100 if price_raw else None,
                "change_pct": float(change_raw) / 100 if change_raw else None,
                "country": country,
            }
        )
    return rows


def _load_marketcap_universe(limit: int = 1000, refresh: bool = False) -> list[dict[str, Any]]:
    if MARKETCAP_CACHE.exists() and not refresh:
        try:
            payload = json.loads(MARKETCAP_CACHE.read_text(encoding="utf-8"))
            fetched_at = _parse_utc_datetime(payload.get("fetched_at"))
            if fetched_at and (_utc_now() - fetched_at).total_seconds() < 12 * 3600:
                return payload.get("rows", [])[:limit]
        except Exception:
            pass

    rows: list[dict[str, Any]] = []
    for page in range(1, MARKETCAP_MAX_PAGES + 1):
        try:
            rows.extend(_fetch_marketcap_page(page))
        except Exception as exc:
            logger.warning("marketcap page %s failed: %s", page, exc)
    us_rows = [r for r in rows if "USA" in str(r.get("country", ""))]
    us_rows = us_rows[:limit]
    payload = {"fetched_at": _utc_now().isoformat(), "rows": us_rows}
    MARKETCAP_CACHE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return us_rows


def _is_optionable_static(info: Any) -> bool:
    derivatives = getattr(info, "stock_derivatives", []) or []
    return any("Option" in str(item) for item in derivatives)


def _marketcap_ticker_score(row: dict[str, Any], quote: Any | None = None, profile_bias: dict[str, float] | None = None) -> float:
    market_cap = float(row.get("market_cap") or 0)
    change_pct = abs(float(row.get("change_pct") or 0))
    volume = float(getattr(quote, "volume", 0) or 0) if quote is not None else 0
    turnover = float(getattr(quote, "turnover", 0) or 0) if quote is not None else 0
    bias = profile_bias or {}
    direction_bias = bias.get("direction_bias", 0.0)
    iv_bias = bias.get("iv_bias", 0.0)
    dte_bias = bias.get("dte_bias", 0.0)
    base = math.log10(max(market_cap, 1.0))
    liq = math.log10(volume + 1.0) * 0.5 + math.log10(turnover + 1.0) * 0.2
    motion = change_pct * 0.8
    profile = direction_bias * 0.1 + iv_bias * 0.05 + dte_bias * 0.03
    return base * 3.0 + liq + motion + profile


def _library_style_bonus(plan: dict[str, Any], conclusion: dict[str, Any]) -> float:
    bonus = 0.0
    if not plan:
        return bonus

    direction = str(conclusion.get("direction", "neutral"))
    dte_style = str(conclusion.get("dte_style", "mixed"))
    iv_style = str(conclusion.get("iv_style", "balanced"))
    structure = str(conclusion.get("structure", "both"))
    entry_style = str(conclusion.get("entry_style", "range_or_confirmation"))
    risk_style = str(conclusion.get("risk_style", "balanced"))
    structure_hint = str(conclusion.get("structure_hint", ""))

    plan_type = str(plan.get("type", "")).lower()
    dte = int(plan.get("dte") or 0)
    iv_pct = plan.get("iv_pct")
    try:
        iv_pct = float(iv_pct) if iv_pct is not None else None
    except Exception:
        iv_pct = None

    if direction == "bullish" and plan_type == "call":
        bonus += 8
    elif direction == "bearish" and plan_type == "put":
        bonus += 8

    if structure in ("call", "put") and plan_type == structure:
        bonus += 4

    if dte_style == "short" and dte <= 14:
        bonus += 4
    elif dte_style == "mid" and 15 <= dte <= 35:
        bonus += 4
    elif dte_style == "long" and dte > 35:
        bonus += 4

    if iv_pct is not None:
        if iv_style == "high" and iv_pct >= 45:
            bonus += 4
        elif iv_style == "low" and iv_pct <= 30:
            bonus += 4
        elif iv_style == "balanced" and 20 <= iv_pct <= 55:
            bonus += 2

    rr = plan.get("risk_reward")
    if rr is not None:
        try:
            rr = float(rr)
            bonus += min(4.0, max(0.0, rr - 1.0) * 1.5)
        except Exception:
            pass

    if entry_style == "breakout_or_pullback_long" and plan_type == "call":
        bonus += 2.5
    elif entry_style == "breakdown_or_rally_short" and plan_type == "put":
        bonus += 2.5
    elif entry_style == "range_or_confirmation" and rr is not None and rr >= 1.8:
        bonus += 1.5

    if risk_style == "tight_stop" and dte <= 14:
        bonus += 1.5
    elif risk_style == "fast_exit" and dte <= 10:
        bonus += 1.5
    elif risk_style == "balanced" and 10 <= dte <= 35:
        bonus += 1.0

    if "spread" in structure_hint and plan.get("spread_pct") is not None:
        try:
            if float(plan.get("spread_pct") or 0) <= 8:
                bonus += 1.0
        except Exception:
            pass

    return round(bonus, 2)


@lru_cache(maxsize=256)
def _cached_company_profile(symbol: str) -> dict[str, Any]:
    try:
        return get_company_profile(_to_yfinance_symbol(symbol))
    except Exception as exc:
        logger.debug("%s company profile failed: %s", symbol, exc)
        return {}


@lru_cache(maxsize=256)
def _cached_recent_news(symbol: str, limit: int = 3) -> list[dict[str, Any]]:
    try:
        return get_recent_news(_to_yfinance_symbol(symbol), limit=limit)
    except Exception as exc:
        logger.debug("%s recent news failed: %s", symbol, exc)
        return []


def _tech_bias_cn(value: str) -> str:
    mapping = {
        "bullish_breakout": "强势突破",
        "mild_bullish": "偏强",
        "bearish_breakdown": "弱势跌破",
        "mild_bearish": "偏弱",
        "overbought": "超买",
        "oversold": "超卖",
        "neutral": "震荡",
    }
    return mapping.get(str(value or "").lower(), str(value or "中性"))


def _iv_rank_cn(iv_rank: Any) -> str:
    try:
        value = float(iv_rank)
    except Exception:
        return "IVR 未知"
    if value >= 70:
        return "波动率偏高"
    if value <= 30:
        return "波动率偏低"
    return "波动率中性"


def _top50_reason_cn(
    row: dict[str, Any],
    plan: dict[str, Any],
    conclusion: dict[str, Any],
    style_bonus: float,
    sim_bonus: float = 0.0,
    profile: dict[str, Any] | None = None,
    news_items: list[dict[str, Any]] | None = None,
) -> str:
    profile = profile or {}
    news_items = news_items or []
    symbol = str(row.get("symbol") or "")
    tech_bias = _tech_bias_cn(row.get("tech_bias"))
    direction = str(conclusion.get("direction", "neutral"))
    dte_style = str(conclusion.get("dte_style", "mixed"))
    iv_style = str(conclusion.get("iv_style", "balanced"))
    entry_style = str(conclusion.get("entry_style", "range_or_confirmation"))
    structure = str(conclusion.get("structure", "both"))
    sector = str(profile.get("sector") or "").strip()
    industry = str(profile.get("industry") or "").strip()
    name = str(profile.get("long_name") or row.get("name") or symbol).strip()

    parts: list[str] = []
    if sector:
        if industry:
            parts.append(f"{sector} / {industry}")
        else:
            parts.append(sector)

    if direction == "bullish" and str(plan.get("type", "")).upper() == "CALL":
        parts.append("与你当前经验库的看涨倾向一致")
    elif direction == "bearish" and str(plan.get("type", "")).upper() == "PUT":
        parts.append("与你当前经验库的看跌倾向一致")
    elif structure == "both":
        parts.append("方向不极端，适合双向里挑更优结构")

    parts.append(f"技术面：{tech_bias}")

    iv_rank = row.get("iv_rank")
    iv_rank_text = "—"
    try:
        if iv_rank is not None:
            iv_rank_text = f"{float(iv_rank):.1f}"
    except Exception:
        iv_rank_text = "—"
    parts.append(f"{_iv_rank_cn(iv_rank)}，IVR {iv_rank_text}")

    dte = plan.get("dte")
    if dte is not None:
        parts.append(f"DTE {dte}，更贴合{dte_style}偏好")

    rr = plan.get("risk_reward")
    if rr is not None:
        try:
            parts.append(f"风险收益比 {float(rr):.2f}")
        except Exception:
            parts.append("风险收益比 —")

    if style_bonus:
        parts.append(f"画像匹配加分 {float(style_bonus):.2f}")
    if sim_bonus:
        parts.append(f"模拟仓学习加分 {float(sim_bonus):.2f}")
    learning_bonus = row.get("learning_bonus")
    if learning_bonus:
        try:
            parts.append(f"自主学习加分 {float(learning_bonus):.2f}")
        except Exception:
            pass

    pre_score = row.get("pre_score")
    if pre_score is not None:
        try:
            parts.append(f"基础流动性/规模评分 {float(pre_score):.2f}")
        except Exception:
            parts.append("基础流动性/规模评分 —")

    if entry_style == "breakout_or_pullback_long" and str(plan.get("type", "")).upper() == "CALL":
        parts.append("更像突破/回踩后的顺势单")
    elif entry_style == "breakdown_or_rally_short" and str(plan.get("type", "")).upper() == "PUT":
        parts.append("更像破位/反弹后的顺势单")
    else:
        parts.append("结构更偏确认后入场")

    if news_items:
        headline = str(news_items[0].get("title") or "").strip()
        if headline:
            parts.append(f"近期新闻：{headline}")

    # 保留简洁，但给出明确的结论句。
    return f"{name}：{'；'.join(parts)}"


def _marketcap_to_output_symbol(ticker: str) -> str:
    return _to_yfinance_symbol(ticker)


def _candles_to_df(candles) -> pd.DataFrame:
    if not candles:
        return _empty_df()
    rows = []
    for c in candles:
        ts = pd.to_datetime(getattr(c, "timestamp", None), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append(
            {
                "t": ts.tz_convert(None),
                "Open": float(getattr(c, "open", np.nan)),
                "High": float(getattr(c, "high", np.nan)),
                "Low": float(getattr(c, "low", np.nan)),
                "Close": float(getattr(c, "close", np.nan)),
                "Volume": float(getattr(c, "volume", np.nan)),
                "Turnover": float(getattr(c, "turnover", np.nan)),
            }
        )
    if not rows:
        return _empty_df()
    df = pd.DataFrame(rows).set_index("t").sort_index()
    return df


def _longbridge_history(symbol: str, count: int) -> pd.DataFrame:
    ctx = _lb_ctx()
    if ctx is None or Period is None or AdjustType is None:
        return _empty_df()
    try:
        candles = ctx.candlesticks(_lb_underlying_symbol(symbol), Period.Day, count, AdjustType.NoAdjust)
        return _candles_to_df(candles)
    except Exception as exc:
        logger.warning("Longbridge history failed for %s: %s", symbol, exc)
        return _empty_df()


def _longbridge_intraday(symbol: str, count: int = 120) -> pd.DataFrame:
    ctx = _lb_ctx()
    if ctx is None or Period is None or AdjustType is None:
        return _empty_df()
    try:
        candles = ctx.candlesticks(_lb_underlying_symbol(symbol), Period.Min_5, count, AdjustType.NoAdjust)
        df = _candles_to_df(candles)
        if df.empty:
            return df
        today = dt.datetime.now(ET).date()
        local_idx = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        df = df.copy()
        df.index = local_idx.tz_localize(None)
        return df[df.index.date == today]
    except Exception as exc:
        logger.warning("Longbridge intraday failed for %s: %s", symbol, exc)
        return _empty_df()


def _longbridge_option_candidates(
    symbol: str,
    spot: float,
    min_dte: int,
    max_dte: int,
    opt_filter: str,
    otm_range: float,
    enrich_quotes: bool = True,
) -> pd.DataFrame:
    ctx = _lb_ctx()
    if ctx is None:
        return _empty_df()
    try:
        expiries = ctx.option_chain_expiry_date_list(_lb_underlying_symbol(symbol))
    except Exception as exc:
        logger.warning("Longbridge expiry list failed for %s: %s", symbol, exc)
        return _empty_df()

    today = dt.date.today()
    rows = []
    lo = spot * (1 - otm_range)
    hi = spot * (1 + otm_range)
    for exp_date in expiries:
        dte = (exp_date - today).days
        if dte < min_dte or dte > max_dte:
            continue
        try:
            chain = ctx.option_chain_info_by_date(_lb_underlying_symbol(symbol), exp_date)
        except Exception as exc:
            logger.warning("Longbridge chain failed for %s %s: %s", symbol, exp_date, exc)
            continue
        for info in chain:
            strike = float(getattr(info, "price", np.nan))
            if not (lo <= strike <= hi):
                continue
            if opt_filter in ("call", "both"):
                rows.append(
                    {
                        "strike": strike,
                        "optionType": "call",
                        "expiry": exp_date.isoformat(),
                        "dte": dte,
                        "lb_symbol": getattr(info, "call_symbol", None),
                    }
                )
            if opt_filter in ("put", "both"):
                rows.append(
                    {
                        "strike": strike,
                        "optionType": "put",
                        "expiry": exp_date.isoformat(),
                        "dte": dte,
                        "lb_symbol": getattr(info, "put_symbol", None),
                    }
                )

    if not rows:
        return _empty_df()

    full = pd.DataFrame(rows)
    if enrich_quotes and len(full) > LONGBRIDGE_OPTION_QUOTE_CHUNK * 2:
        logger.info("Longbridge option chain too wide for %s (%s rows), skipping bulk quote enrichment", symbol, len(full))
        enrich_quotes = False
    if enrich_quotes:
        quotes = _longbridge_option_quotes([s for s in full["lb_symbol"].dropna().unique().tolist() if s])
        last_values = []
        volume_values = []
        oi_values = []
        iv_values = []
        status_values = []

        for _, row in full.iterrows():
            q = quotes.get(row["lb_symbol"])
            if q is None:
                last_values.append(np.nan)
                volume_values.append(np.nan)
                oi_values.append(np.nan)
                iv_values.append(np.nan)
                status_values.append(None)
                continue
            last_values.append(_safe(getattr(q, "last_done", None)))
            volume_values.append(int(getattr(q, "volume", 0) or 0))
            oi_values.append(int(getattr(q, "open_interest", 0) or 0))
            iv = getattr(q, "implied_volatility", None)
            iv_values.append(_safe(iv * 100) if iv is not None else np.nan)
            status_values.append(getattr(q, "trade_status", None))

        full["last"] = last_values
        full["volume"] = volume_values
        full["oi"] = oi_values
        full["iv_pct"] = iv_values
        full["trade_status"] = status_values
    else:
        full["last"] = np.nan
        full["volume"] = np.nan
        full["oi"] = np.nan
        full["iv_pct"] = np.nan
        full["trade_status"] = None

    full["moneyness"] = ((full["strike"] - spot) / spot * 100).round(2)
    full["score"] = (
        full["volume"].fillna(0) * 0.4
        + full["oi"].fillna(0) * 0.4
        + full["iv_pct"].between(15, 50).astype(int) * 20
        + (1 / (full["moneyness"].abs().fillna(999) + 1)) * 20
    )
    return full


def _longbridge_quote(symbol: str) -> Optional[dict]:
    ctx = _lb_ctx()
    if ctx is None:
        return None
    try:
        quote_list = ctx.quote([_lb_underlying_symbol(symbol)])
        if not quote_list:
            return None
        q = quote_list[0]
        price = getattr(q, "last_done", None) or getattr(q, "prev_close", None) or getattr(q, "open", None)
        return {
            "symbol": _normalize_symbol(symbol),
            "price": _safe(price),
            "open": _safe(getattr(q, "open", None)),
            "high": _safe(getattr(q, "high", None)),
            "low": _safe(getattr(q, "low", None)),
            "prev_close": _safe(getattr(q, "prev_close", None)),
            "volume": int(getattr(q, "volume", 0) or 0),
            "turnover": _safe(getattr(q, "turnover", None), 3),
            "timestamp": getattr(q, "timestamp", None),
            "source": "longbridge",
        }
    except Exception as exc:
        logger.warning("Longbridge quote failed for %s: %s", symbol, exc)
        return None


def _webull_quote(symbol: str, config: dict[str, Any] | None = None) -> Optional[dict]:
    cfg = _merge_webull_config(config or _load_webull_config())
    try:
        _, data_client = _webull_data_client(cfg)
        payload = _webull_response_payload(data_client.market_data.get_snapshot(_symbol_root(symbol), WebullCategory.US_STOCK.name))
        row = _webull_snapshot_row(payload)
        if not row:
            return None
        quote = _webull_price_from_row(row)
        if not quote.get("symbol"):
            quote["symbol"] = _normalize_symbol(symbol)
        return quote
    except Exception as exc:
        logger.warning("Webull quote failed for %s: %s", symbol, exc)
        return None


def _webull_history(symbol: str, count: int, timespan: Any = None, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = _merge_webull_config(config or _load_webull_config())
    bar_span = timespan or (WebullTimespan.D.name if WebullTimespan is not None else "D")
    try:
        _, data_client = _webull_data_client(cfg)
        payload = _webull_response_payload(
            data_client.market_data.get_history_bar(
                _symbol_root(symbol),
                WebullCategory.US_STOCK.name,
                bar_span,
                count=str(min(max(int(count), 1), 1200)),
            )
        )
        return _webull_bars_to_df(payload)
    except Exception as exc:
        logger.warning("Webull history failed for %s (%s): %s", symbol, bar_span, exc)
        return _empty_df()


def _webull_market_probe(config: dict[str, Any], symbol: str = "AAPL") -> dict[str, Any]:
    result = {
        "symbol": _symbol_root(symbol),
        "sdk_available": WebullDataClient is not None,
        "available": False,
        "permission": "unknown",
        "instrument_ok": False,
        "quote_ok": False,
        "message": "",
    }
    if WebullDataClient is None:
        result["message"] = "webull data sdk unavailable"
        return result
    try:
        _, data_client = _webull_data_client(config)
        payload = _webull_response_payload(data_client.instrument.get_instrument(symbols=_symbol_root(symbol), category=WebullCategory.US_STOCK.name))
        instrument_data = payload.get("data") if isinstance(payload, dict) else None
        result["instrument_ok"] = bool(instrument_data)
        try:
            snap_payload = _webull_response_payload(data_client.market_data.get_snapshot(_symbol_root(symbol), WebullCategory.US_STOCK.name))
            row = _webull_snapshot_row(snap_payload)
            if row:
                result["available"] = True
                result["permission"] = "subscribed"
                result["quote_ok"] = True
                result["message"] = "webull market data available"
            else:
                result["message"] = "webull snapshot empty"
        except Exception as exc:
            result["permission"] = _webull_market_permission(str(exc))
            result["message"] = str(exc)
    except Exception as exc:
        result["permission"] = _webull_market_permission(str(exc))
        result["message"] = str(exc)
    return result


def _longbridge_market_temperature(market: str = "US") -> Optional[dict]:
    ctx = _lb_ctx()
    if ctx is None:
        return None
    enum_market = _lb_market_enum(market)
    if enum_market is None:
        return None
    try:
        temp = ctx.market_temperature(enum_market)
        return {
            "market": market.upper(),
            "temperature": getattr(temp, "temperature", None),
            "description": getattr(temp, "description", None),
            "valuation": getattr(temp, "valuation", None),
            "sentiment": getattr(temp, "sentiment", None),
            "timestamp": getattr(temp, "timestamp", None),
            "source": "longbridge",
        }
    except Exception as exc:
        logger.warning("Longbridge market temperature failed: %s", exc)
        return None


def _longbridge_option_quotes(symbols: list[str]) -> dict[str, Any]:
    ctx = _lb_ctx()
    if ctx is None or not symbols:
        return {}
    unique_symbols = []
    seen = set()
    for symbol in symbols:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        unique_symbols.append(symbol)

    out: dict[str, Any] = {}
    for chunk in _chunked(unique_symbols, LONGBRIDGE_OPTION_QUOTE_CHUNK):
        try:
            quotes = ctx.option_quote(chunk)
            for quote in quotes:
                key = getattr(quote, "symbol", "")
                if key:
                    out[key] = quote
        except Exception as exc:
            logger.warning("Longbridge option_quote failed for chunk(size=%s): %s", len(chunk), exc)
    return out


def _longbridge_option_depth(symbol: str) -> tuple[Optional[float], Optional[float]]:
    ctx = _lb_ctx()
    if ctx is None:
        return None, None
    try:
        depth = ctx.depth(symbol)
        bid = None
        ask = None
        if getattr(depth, "bids", None):
            bid = getattr(depth.bids[0], "price", None)
        if getattr(depth, "asks", None):
            ask = getattr(depth.asks[0], "price", None)
        return _safe(bid), _safe(ask)
    except Exception as exc:
        logger.warning("Longbridge depth failed for %s: %s", symbol, exc)
        return None, None


def _chunked(seq: list[str], size: int = 80):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _now_et_iso() -> str:
    return dt.datetime.now(ET).isoformat()


def _iso_timestamp(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _parse_iso_datetime(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return ET.localize(parsed)
    return parsed.astimezone(ET)


def _days_since(value: Any, now: Optional[dt.datetime] = None) -> Optional[float]:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    ref = now or dt.datetime.now(ET)
    return max(0.0, (ref - parsed).total_seconds() / 86400.0)


def _recency_multiplier(value: Any, half_life_days: float = 21.0, floor: float = 0.35, now: Optional[dt.datetime] = None) -> float:
    age_days = _days_since(value, now=now)
    if age_days is None:
        return 1.0
    decay = 0.5 ** (age_days / max(1.0, half_life_days))
    return round(max(floor, min(1.0, decay)), 3)


def _within_days(value: Any, days: float, now: Optional[dt.datetime] = None) -> bool:
    age_days = _days_since(value, now=now)
    return age_days is not None and age_days <= max(0.0, float(days))


def _bucket_market_temperature(value: Any) -> str:
    temp = _num(value, 50.0)
    if temp >= 75:
        return "hot"
    if temp >= 60:
        return "warm"
    if temp <= 30:
        return "cold"
    if temp <= 45:
        return "cool"
    return "neutral"


def _bucket_market_sentiment(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("fear", "panic", "bear")):
        return "fear"
    if any(token in text for token in ("greed", "euphoria", "bull")):
        return "greed"
    return "neutral"


def _bucket_market_valuation(value: Any) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("cheap", "low", "underval")):
        return "cheap"
    if any(token in text for token in ("expensive", "high", "overval")):
        return "rich"
    return "fair"


def _load_market_environment_cache(max_age_minutes: float = 30.0) -> Optional[dict[str, Any]]:
    if not MARKET_ENV_CACHE.exists():
        return None
    try:
        payload = json.loads(MARKET_ENV_CACHE.read_text(encoding="utf-8"))
        fetched_at = _parse_utc_datetime(payload.get("fetched_at"))
        if fetched_at is None:
            return None
        age_minutes = (_utc_now() - fetched_at).total_seconds() / 60.0
        if age_minutes > max_age_minutes:
            return None
        if str(payload.get("trend_bucket") or "").upper() in {"CALL", "PUT"}:
            return None
        return payload
    except Exception:
        return None


def _save_market_environment_cache(payload: dict[str, Any]) -> None:
    try:
        MARKET_ENV_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_iso_timestamp), encoding="utf-8")
    except Exception as exc:
        logger.debug("market environment cache save skipped: %s", exc)


def _current_market_environment(refresh: bool = False) -> dict[str, Any]:
    cached = None if refresh else _load_market_environment_cache(max_age_minutes=30.0)
    if cached:
        return cached
    temperature = _longbridge_market_temperature("US") or {}
    spy_hist, source = _daily_history("SPY", 40)
    trend_bucket = "neutral"
    spy_rsi = None
    if not spy_hist.empty and "Close" in spy_hist:
        close = spy_hist["Close"].dropna()
        if len(close) >= 20:
            sma20 = _safe(close.tail(20).mean())
            sma50 = _safe(close.tail(40).mean()) if len(close) >= 40 else sma20
            spy_rsi = _safe(_calc_rsi(close)) if len(close) >= 15 else None
            tech = _tech_signal(float(close.iloc[-1]), sma20, sma50, spy_rsi)
            trend_code = _tech_direction_from_bias(tech) or "neutral"
            if str(trend_code).upper() == "CALL":
                trend_bucket = "bullish"
            elif str(trend_code).upper() == "PUT":
                trend_bucket = "bearish"
            else:
                trend_bucket = "neutral"
    payload = {
        "fetched_at": _utc_now().isoformat(),
        "timestamp": _now_et_iso(),
        "market": str(temperature.get("market") or "US"),
        "temperature": _safe(temperature.get("temperature")),
        "description": temperature.get("description"),
        "valuation": temperature.get("valuation"),
        "sentiment": temperature.get("sentiment"),
        "temp_bucket": _bucket_market_temperature(temperature.get("temperature")),
        "sentiment_bucket": _bucket_market_sentiment(temperature.get("sentiment") or temperature.get("description")),
        "valuation_bucket": _bucket_market_valuation(temperature.get("valuation")),
        "trend_bucket": trend_bucket,
        "trend_source": source,
        "spy_rsi": spy_rsi,
    }
    payload["label"] = f"{payload['temp_bucket']}_{payload['sentiment_bucket']}_{payload['trend_bucket']}"
    _save_market_environment_cache(payload)
    return payload


def _default_sim_state() -> dict[str, Any]:
    return {
        "updated_at": None,
        "trades": [],
        "closed": [],
    }


def _default_learning_state() -> dict[str, Any]:
    return {
        "updated_at": None,
        "closed_count": 0,
        "open_position_count": 0,
        "open_position_sample_weight": 0.0,
        "resolved_signal_count": 0,
        "confidence": 0.0,
        "summary": "样本不足，暂未形成稳定学习结论",
        "preferences": {
            "direction": "balanced",
            "dte": "balanced",
            "iv": "balanced",
            "rr": "balanced",
        },
        "weights": {
            "type_call": 0.0,
            "type_put": 0.0,
            "dte_short": 0.0,
            "dte_mid": 0.0,
            "dte_long": 0.0,
            "iv_low": 0.0,
            "iv_mid": 0.0,
            "iv_high": 0.0,
            "rr_low": 0.0,
            "rr_mid": 0.0,
            "rr_high": 0.0,
        },
        "factor_weights": {
            "ivrv_cheap": 0.0,
            "ivrv_neutral": 0.0,
            "ivrv_rich": 0.0,
            "skew_supportive": 0.0,
            "skew_neutral": 0.0,
            "skew_adverse": 0.0,
            "flow_strong": 0.0,
            "flow_normal": 0.0,
            "flow_weak": 0.0,
            "liquidity_tight": 0.0,
            "liquidity_fair": 0.0,
            "liquidity_wide": 0.0,
        },
        "source_weights": {
            "scan": 0.0,
            "unusual": 0.0,
            "manual": 0.0,
            "pool": 0.0,
            "webull": 0.0,
            "other": 0.0,
        },
        "stats": {},
        "sim_stats": {},
        "signal_stats": {},
        "source_stats": {},
        "environment_weights": {
            "temp_hot": 0.0,
            "temp_warm": 0.0,
            "temp_neutral": 0.0,
            "temp_cool": 0.0,
            "temp_cold": 0.0,
            "sentiment_greed": 0.0,
            "sentiment_neutral": 0.0,
            "sentiment_fear": 0.0,
            "valuation_rich": 0.0,
            "valuation_fair": 0.0,
            "valuation_cheap": 0.0,
            "trend_bullish": 0.0,
            "trend_neutral": 0.0,
            "trend_bearish": 0.0,
        },
        "environment_stats": {},
        "knowledge_summary": "知识库尚未编译",
        "knowledge_bias": {},
        "knowledge_files": {},
    }


def _default_signal_memory() -> dict[str, Any]:
    return {
        "updated_at": None,
        "signals": [],
    }


def _load_sim_state() -> dict[str, Any]:
    with SIM_STATE_LOCK:
        if not SIM_TRADING_STATE.exists():
            return _default_sim_state()
        try:
            payload = json.loads(SIM_TRADING_STATE.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return _default_sim_state()
            payload.setdefault("trades", [])
            payload.setdefault("closed", [])
            payload.setdefault("updated_at", None)
            return payload
        except Exception as exc:
            logger.warning("sim state load failed: %s", exc)
            return _default_sim_state()


def _save_sim_state(state: dict[str, Any]) -> None:
    with SIM_STATE_LOCK:
        SIM_TRADING_STATE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=_iso_timestamp),
            encoding="utf-8",
        )


def _load_learning_state() -> dict[str, Any]:
    if not LEARNING_STATE.exists():
        return _default_learning_state()
    try:
        payload = json.loads(LEARNING_STATE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return _default_learning_state()
        base = _default_learning_state()
        base.update(payload)
        base["preferences"] = {**_default_learning_state()["preferences"], **(payload.get("preferences") or {})}
        base["weights"] = {**_default_learning_state()["weights"], **(payload.get("weights") or {})}
        base["factor_weights"] = {**_default_learning_state()["factor_weights"], **(payload.get("factor_weights") or {})}
        base["source_weights"] = {**_default_learning_state()["source_weights"], **(payload.get("source_weights") or {})}
        base["environment_weights"] = {**_default_learning_state()["environment_weights"], **(payload.get("environment_weights") or {})}
        return base
    except Exception as exc:
        logger.warning("learning state load failed: %s", exc)
        return _default_learning_state()


def _save_learning_state(state: dict[str, Any]) -> None:
    LEARNING_STATE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=_iso_timestamp),
        encoding="utf-8",
    )


def _load_signal_memory() -> dict[str, Any]:
    if not SIGNAL_MEMORY.exists():
        return _default_signal_memory()
    try:
        payload = json.loads(SIGNAL_MEMORY.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return _default_signal_memory()
        payload.setdefault("signals", [])
        payload.setdefault("updated_at", None)
        return payload
    except Exception as exc:
        logger.warning("signal memory load failed: %s", exc)
        return _default_signal_memory()


def _save_signal_memory(state: dict[str, Any]) -> None:
    SIGNAL_MEMORY.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=_iso_timestamp),
        encoding="utf-8",
    )


def _knowledge_file_map() -> dict[str, Path]:
    return {
        "index": KNOWLEDGE_WIKI_INDEX,
        "factors": KNOWLEDGE_WIKI_FACTORS,
        "sources": KNOWLEDGE_WIKI_SOURCES,
        "signals": KNOWLEDGE_WIKI_SIGNALS,
        "reports": KNOWLEDGE_WIKI_REPORTS,
        "learning": KNOWLEDGE_WIKI_LEARNING,
        "market": KNOWLEDGE_WIKI_MARKET,
    }


def _knowledge_files_payload() -> dict[str, str]:
    return {key: str(path) for key, path in _knowledge_file_map().items()}


def _ranked_bucket_items(mapping: dict[str, Any], top_n: int = 3, min_abs_score: float = 0.25) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = []
    for key, value in (mapping or {}).items():
        score = _num(value if not isinstance(value, dict) else value.get("score"))
        if abs(score) < min_abs_score:
            continue
        items.append({"key": str(key), "score": round(score, 2)})
    items.sort(key=lambda item: item["score"], reverse=True)
    positives = [item for item in items if item["score"] > 0][:top_n]
    negatives = [item for item in sorted(items, key=lambda item: item["score"]) if item["score"] < 0][:top_n]
    return positives, negatives


def _knowledge_bias_from_learning(payload: dict[str, Any]) -> dict[str, Any]:
    confidence = _num(payload.get("confidence"))
    weights = payload.get("weights", {}) or {}
    factor_weights = payload.get("factor_weights", {}) or {}
    source_weights = payload.get("source_weights", {}) or {}
    environment_weights = payload.get("environment_weights", {}) or {}
    preferences = payload.get("preferences", {}) or {}
    recent_stats = payload.get("recent_stats", {}) or {}
    favored_factors, weak_factors = _ranked_bucket_items(factor_weights, top_n=4, min_abs_score=0.35)
    favored_sources, weak_sources = _ranked_bucket_items(source_weights, top_n=3, min_abs_score=0.3)
    favored_env, weak_env = _ranked_bucket_items(environment_weights, top_n=4, min_abs_score=0.25)
    rules: list[str] = []
    if confidence >= 0.55:
        direction = preferences.get("direction")
        if direction == "call" and _num(weights.get("type_call")) > 0.35:
            rules.append("高置信度样本更支持顺势 CALL")
        elif direction == "put" and _num(weights.get("type_put")) > 0.35:
            rules.append("高置信度样本更支持顺势 PUT")
        dte_pref = preferences.get("dte")
        if dte_pref in {"short", "mid", "long"}:
            rules.append(f"DTE 更偏向 {dte_pref}")
        iv_pref = preferences.get("iv")
        if iv_pref in {"low", "mid", "high"}:
            rules.append(f"IV 条件更偏向 {iv_pref}")
    if recent_stats.get("samples_7d", 0) >= 2:
        rules.append(f"近7天方向偏好: {recent_stats.get('direction_7d', 'balanced')}")
    if recent_stats.get("samples_30d", 0) >= 3:
        rules.append(f"近30天方向偏好: {recent_stats.get('direction_30d', 'balanced')}")
    if favored_sources:
        rules.append("优先来源: " + ", ".join(item["key"] for item in favored_sources))
    if weak_sources:
        rules.append("谨慎来源: " + ", ".join(item["key"] for item in weak_sources))
    if favored_env:
        rules.append("优先环境: " + ", ".join(item["key"] for item in favored_env[:3]))
    if weak_env:
        rules.append("规避环境: " + ", ".join(item["key"] for item in weak_env[:3]))
    if favored_factors:
        rules.append("优先因子: " + ", ".join(item["key"] for item in favored_factors[:3]))
    if weak_factors:
        rules.append("规避因子: " + ", ".join(item["key"] for item in weak_factors[:3]))
    summary = "；".join(rules) if rules else "知识库暂未形成足够强的显式规则"
    return {
        "summary": summary,
        "confidence_gate": round(confidence, 2),
        "preferred_direction": preferences.get("direction", "balanced"),
        "preferred_dte": preferences.get("dte", "balanced"),
        "preferred_iv": preferences.get("iv", "balanced"),
        "recent_direction_7d": recent_stats.get("direction_7d", "balanced"),
        "recent_direction_30d": recent_stats.get("direction_30d", "balanced"),
        "favored_sources": favored_sources,
        "weak_sources": weak_sources,
        "favored_environments": favored_env,
        "weak_environments": weak_env,
        "favored_factors": favored_factors,
        "weak_factors": weak_factors,
        "rules": rules,
    }


def _knowledge_report_lines(payload: dict[str, Any]) -> list[str]:
    sim_stats = payload.get("sim_stats", {}) or {}
    signal_stats = payload.get("signal_stats", {}) or {}
    stats = payload.get("stats", {}) or {}
    recent_stats = payload.get("recent_stats", {}) or {}
    lines = [
        f"- 更新时间: {payload.get('updated_at') or 'N/A'}",
        f"- 学习摘要: {payload.get('summary') or 'N/A'}",
        f"- 知识摘要: {(payload.get('knowledge_bias') or {}).get('summary') or 'N/A'}",
        f"- 平仓样本: {payload.get('closed_count', 0)}",
        f"- 开放仓位: {payload.get('open_position_count', 0)}",
        f"- 跟踪信号: {payload.get('resolved_signal_count', 0)}",
        f"- 开放仓平均收益: {sim_stats.get('open_avg_return_pct', 0)}%",
        f"- 信号胜率: {signal_stats.get('signal_win_rate', 'N/A')}",
    ]
    if recent_stats:
        lines.append(
            f"- 近7天: samples {recent_stats.get('samples_7d', 0)} / resolved {recent_stats.get('resolved_7d', 0)} / bias {recent_stats.get('direction_7d', 'balanced')}"
        )
        lines.append(
            f"- 近30天: samples {recent_stats.get('samples_30d', 0)} / resolved {recent_stats.get('resolved_30d', 0)} / bias {recent_stats.get('direction_30d', 'balanced')}"
        )
    top_symbols = stats.get("top_symbols") or []
    if top_symbols:
        lines.append("")
        lines.append("## Top Symbols")
        for item in top_symbols[:5]:
            lines.append(f"- {item.get('symbol')}: pnl {item.get('pnl')} / win_rate {item.get('win_rate')}% / count {item.get('count')}")
    return lines


def _write_learning_knowledge_wiki(payload: dict[str, Any], resolved_signals: list[dict[str, Any]]) -> None:
    files = _knowledge_file_map()
    files["index"].write_text(
        "\n".join(
            [
                "# Learning Wiki",
                "",
                "- This local wiki is auto-maintained from simulation trades, resolved signals, source stats, and factor stats.",
                "- Inspired by the workflow described in Andrej Karpathy's `llm-wiki.md`: compile reusable conclusions into markdown, then read them back as working memory.",
                "",
                "## Files",
                f"- [factors.md](./{files['factors'].name})",
                f"- [sources.md](./{files['sources'].name})",
                f"- [signals.md](./{files['signals'].name})",
                f"- [learning.md](./{files['learning'].name})",
                f"- [reports.md](./{files['reports'].name})",
                f"- [market_memory.md](./{files['market'].name})",
                "",
            ]
            + _knowledge_report_lines(payload)
        ),
        encoding="utf-8",
    )

    factor_weights = payload.get("factor_weights", {}) or {}
    favored_factors, weak_factors = _ranked_bucket_items(factor_weights, top_n=6, min_abs_score=0.2)
    files["factors"].write_text(
        "\n".join(
            [
                "# Factor Wiki",
                "",
                "## Favored",
                *([f"- {item['key']}: {item['score']}" for item in favored_factors] or ["- none"]),
                "",
                "## Weak",
                *([f"- {item['key']}: {item['score']}" for item in weak_factors] or ["- none"]),
                "",
                "## Raw",
                *[f"- {key}: {round(_num(value), 2)}" for key, value in sorted(factor_weights.items())],
            ]
        ),
        encoding="utf-8",
    )

    source_stats = payload.get("source_stats", {}) or {}
    favored_sources, weak_sources = _ranked_bucket_items(payload.get("source_weights", {}) or {}, top_n=6, min_abs_score=0.15)
    files["sources"].write_text(
        "\n".join(
            [
                "# Source Wiki",
                "",
                "## Favored",
                *([f"- {item['key']}: {item['score']}" for item in favored_sources] or ["- none"]),
                "",
                "## Weak",
                *([f"- {item['key']}: {item['score']}" for item in weak_sources] or ["- none"]),
                "",
                "## Raw",
                *[
                    f"- {key}: score {round(_num((value or {}).get('score')), 2)}, win_rate {(value or {}).get('win_rate')}, sample_weight {(value or {}).get('sample_weight')}"
                    for key, value in sorted(source_stats.items())
                ],
            ]
        ),
        encoding="utf-8",
    )

    files["signals"].write_text(
        "\n".join(
            [
                "# Signal Wiki",
                "",
                "## Recent Resolved Signals",
                *(
                    [
                        f"- {item.get('symbol')}: status={item.get('status')} edge={_safe(item.get('directional_edge_pct'))}% success={item.get('success')} source={item.get('source_bucket') or item.get('source')}"
                        for item in resolved_signals[:20]
                    ]
                    or ["- none"]
                ),
            ]
        ),
        encoding="utf-8",
    )

    files["learning"].write_text("\n".join(["# Learning Report", "", *_knowledge_report_lines(payload)]), encoding="utf-8")
    if not files["market"].exists():
        files["market"].write_text("# Market Memory\n", encoding="utf-8")


def _knowledge_context_for_prompt(max_chars: int = 2200) -> str:
    profile = _adaptive_learning_profile()
    bias = profile.get("knowledge_bias", {}) or {}
    parts = [
        f"学习摘要: {profile.get('summary') or 'N/A'}",
        f"知识摘要: {profile.get('knowledge_summary') or bias.get('summary') or 'N/A'}",
    ]
    rules = bias.get("rules") or []
    if rules:
        parts.append("知识规则: " + "；".join(str(rule) for rule in rules[:6]))
    learning_path = _knowledge_file_map()["learning"]
    reports_path = _knowledge_file_map()["reports"]
    market_path = _knowledge_file_map()["market"]
    try:
        report_text = learning_path.read_text(encoding="utf-8")
    except Exception:
        report_text = ""
    if report_text:
        tail = report_text[-min(len(report_text), 1200):].strip()
        if tail:
            parts.append("近期报告摘录:\n" + tail)
    try:
        session_text = reports_path.read_text(encoding="utf-8-sig")
    except Exception:
        session_text = ""
    if session_text:
        tail = session_text[-min(len(session_text), 900):].strip()
        if tail:
            parts.append("近期会话摘要:\n" + tail)
    try:
        market_text = market_path.read_text(encoding="utf-8-sig")
    except Exception:
        market_text = ""
    if market_text:
        tail = market_text[-min(len(market_text), 900):].strip()
        if tail:
            parts.append("近期市场记忆:\n" + tail)
    text = "\n".join(parts).strip()
    return text[:max_chars]


def _append_session_report_to_knowledge_wiki(session_label: str, report_text: str, timestamp: Optional[str] = None) -> None:
    if not report_text:
        return
    files = _knowledge_file_map()
    ts = str(timestamp or _now_et_iso())
    report_path = files["reports"]
    try:
        existing = report_path.read_text(encoding="utf-8-sig") if report_path.exists() else "# Learning Report\n"
    except Exception:
        existing = "# Learning Report\n"
    if existing.startswith("# Learning Report") or "## Top Symbols" in existing[:800]:
        existing = "# Session Reports\n"
    header = "# Session Reports"
    body = existing
    if body.startswith("# "):
        first_break = body.find("\n")
        body = body[first_break + 1:] if first_break >= 0 else ""
    sections = [s.strip() for s in re.split(r"(?m)^## ", body) if s.strip()]
    sections = [s for s in sections if not s.startswith("# Session Reports")]
    sections.append(_summarize_session_report_for_wiki(session_label, report_text, ts))
    sections = sections[-12:]
    merged = header + "\n\n" + "\n\n".join("## " + section for section in sections) + "\n"
    report_path.write_text(merged, encoding="utf-8")


def _summarize_session_report_for_wiki(session_label: str, report_text: str, timestamp: str) -> str:
    lines = [line.rstrip() for line in str(report_text or "").splitlines()]
    nonempty = [line for line in lines if line.strip()]
    summary: list[str] = [f"{timestamp} | {session_label}"]
    unusual_lines = [line for line in nonempty if "**" in line and "分数" in line]
    candidate_lines = [
        line for line in nonempty
        if (line[:3].strip().rstrip(".").isdigit() or re.match(r"^\d+\.", line)) and line not in unusual_lines
    ]
    bias_line = next((line for line in nonempty if "方向" in line and "DTE" in line and "IV" in line), "")
    market_lines = [line for line in nonempty if line.startswith("> ")][:3]
    if bias_line:
        summary.extend(["", bias_line])
    if market_lines:
        summary.extend(["", "Market"] + market_lines)
    if candidate_lines:
        summary.extend(["", "Top Candidates"] + candidate_lines[:5])
    if unusual_lines:
        summary.extend(["", "Unusual Focus"] + unusual_lines[:5])
    if not candidate_lines and not unusual_lines:
        summary.extend(["", *nonempty[:12]])
    return "\n".join(summary)


def _append_market_memory(section: str, lines: list[str], timestamp: Optional[str] = None) -> None:
    files = _knowledge_file_map()
    path = files["market"]
    ts = str(timestamp or _now_et_iso())
    try:
        existing = path.read_text(encoding="utf-8-sig") if path.exists() else "# Market Memory\n"
    except Exception:
        existing = "# Market Memory\n"
    header = "# Market Memory"
    body = existing
    if body.startswith("# "):
        first_break = body.find("\n")
        body = body[first_break + 1:] if first_break >= 0 else ""
    sections = [s.strip() for s in re.split(r"(?m)^## ", body) if s.strip()]
    sections = [s for s in sections if not s.startswith("# Market Memory")]
    cleaned_lines = [str(line).rstrip() for line in (lines or []) if str(line).strip()]
    if not cleaned_lines:
        return
    sections.append(f"{ts} | {section}\n\n" + "\n".join(cleaned_lines))
    sections = sections[-18:]
    merged = header + "\n\n" + "\n\n".join("## " + section for section in sections) + "\n"
    path.write_text(merged, encoding="utf-8")


def _format_price(value: Any) -> Optional[float]:
    return _safe(value, 4)


def _extract_option_mark_from_quote(q: Any) -> dict[str, Any]:
    if q is None:
        return {}
    bid = getattr(q, "bid", None)
    ask = getattr(q, "ask", None)
    last = getattr(q, "last_done", None)
    mark = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        mark = round((float(bid) + float(ask)) / 2, 4)
    elif last is not None and last > 0:
        mark = _safe(last, 4)
    elif ask is not None and ask > 0:
        mark = _safe(ask, 4)
    elif bid is not None and bid > 0:
        mark = _safe(bid, 4)
    return {
        "mark": mark,
        "bid": _safe(bid, 4),
        "ask": _safe(ask, 4),
        "last": _safe(last, 4),
        "volume": int(getattr(q, "volume", 0) or 0),
        "oi": int(getattr(q, "open_interest", 0) or 0),
        "iv_pct": _safe(getattr(q, "implied_volatility", None) * 100 if getattr(q, "implied_volatility", None) is not None else None),
        "source": "longbridge",
    }


def _get_underlying_mark(symbol: str) -> dict[str, Any]:
    quote = _longbridge_quote(symbol)
    if quote and quote.get("price"):
        return {
            "price": _safe(quote.get("price")),
            "source": quote.get("source", "longbridge"),
            "timestamp": _iso_timestamp(quote.get("timestamp")),
        }
    try:
        hist = _yahoo_history(symbol, period="5d")
        if not hist.empty:
            price = _safe(hist["Close"].iloc[-1])
            return {"price": price, "source": "yahoo", "timestamp": None}
    except Exception as exc:
        logger.debug("%s underlying fallback failed: %s", symbol, exc)
    return {"price": None, "source": None, "timestamp": None}


def _get_option_mark(symbol: str, expiry: str, strike: float, opt_type: str) -> dict[str, Any]:
    lb_symbol = _lb_option_symbol(symbol, expiry, strike, opt_type)
    q = None
    try:
        quotes = _longbridge_option_quotes([lb_symbol])
        q = quotes.get(lb_symbol)
    except Exception as exc:
        logger.debug("%s option quote failed: %s", lb_symbol, exc)
    if q is not None:
        out = _extract_option_mark_from_quote(q)
        out["lb_symbol"] = lb_symbol
        if out.get("mark") is not None:
            return out

    try:
        ticker = _yf_ticker(symbol)
        chain = ticker.option_chain(expiry)
        df = chain.calls if str(opt_type).upper() == "CALL" else chain.puts
        if not df.empty:
            row = df.loc[(df["strike"] - float(strike)).abs().idxmin()]
            bid = row.get("bid")
            ask = row.get("ask")
            last = row.get("lastPrice")
            mark = None
            if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                mark = round((float(bid) + float(ask)) / 2, 4)
            elif pd.notna(last) and last > 0:
                mark = _safe(last, 4)
            elif pd.notna(ask) and ask > 0:
                mark = _safe(ask, 4)
            elif pd.notna(bid) and bid > 0:
                mark = _safe(bid, 4)
            return {
                "mark": mark,
                "bid": _safe(bid, 4),
                "ask": _safe(ask, 4),
                "last": _safe(last, 4),
                "volume": int(row.get("volume", 0) or 0) if pd.notna(row.get("volume")) else 0,
                "oi": int(row.get("openInterest", 0) or 0) if pd.notna(row.get("openInterest")) else 0,
                "iv_pct": _safe(float(row.get("impliedVolatility", 0) or 0) * 100, 2) if pd.notna(row.get("impliedVolatility")) else None,
                "source": "yahoo",
                "lb_symbol": lb_symbol,
            }
    except Exception as exc:
        logger.debug("%s option mark fallback failed: %s", lb_symbol, exc)
    return {"mark": None, "bid": None, "ask": None, "last": None, "volume": 0, "oi": 0, "iv_pct": None, "source": None, "lb_symbol": lb_symbol}


def _summarize_sim_state(state: dict[str, Any]) -> dict[str, Any]:
    trades = state.get("trades", [])
    open_trades = [t for t in trades if t.get("status") == "open"]
    closed = state.get("closed", [])
    open_pnl = sum(float(t.get("unrealized_pnl", 0) or 0) for t in open_trades)
    realized_pnl = sum(float(t.get("realized_pnl", 0) or 0) for t in closed)
    total = open_pnl + realized_pnl
    wins = [t for t in closed if float(t.get("realized_pnl", 0) or 0) > 0]
    win_rate = round(len(wins) / len(closed) * 100, 2) if closed else 0.0
    avg_return = 0.0
    if closed:
        returns = []
        for t in closed:
            entry = float(t.get("entry_price", 0) or 0)
            realized = float(t.get("realized_pnl", 0) or 0)
            qty = float(t.get("qty", 1) or 1)
            if entry > 0 and qty > 0:
                returns.append(realized / (entry * qty * SIM_TRADE_MULTIPLIER) * 100)
        if returns:
            avg_return = round(sum(returns) / len(returns), 2)
    by_type = {}
    for t in closed:
        ttype = str(t.get("option_type") or "").upper() or "UNKNOWN"
        bucket = by_type.setdefault(ttype, {"count": 0, "pnl": 0.0})
        bucket["count"] += 1
        bucket["pnl"] += float(t.get("realized_pnl", 0) or 0)
    for t in open_trades:
        ttype = str(t.get("option_type") or "").upper() or "UNKNOWN"
        bucket = by_type.setdefault(ttype, {"count": 0, "pnl": 0.0})
        bucket["count"] += 1
        bucket["pnl"] += float(t.get("unrealized_pnl", 0) or 0)
    return {
        "open_count": len(open_trades),
        "closed_count": len(closed),
        "open_unrealized_pnl": round(open_pnl, 2),
        "realized_pnl": round(realized_pnl, 2),
        "total_pnl": round(total, 2),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "by_type": by_type,
        "last_updated": state.get("updated_at"),
    }


def _bucket_from_dte(value: Any) -> str:
    try:
        dte = int(value or 0)
    except Exception:
        dte = 0
    if dte <= 14:
        return "short"
    if dte <= 35:
        return "mid"
    return "long"


def _bucket_from_iv(value: Any) -> str:
    try:
        iv = float(value or 0)
    except Exception:
        iv = 0.0
    if iv <= 30:
        return "low"
    if iv <= 60:
        return "mid"
    return "high"


def _bucket_from_rr(value: Any) -> str:
    try:
        rr = float(value or 0)
    except Exception:
        rr = 0.0
    if rr < 1.5:
        return "low"
    if rr <= 2.5:
        return "mid"
    return "high"


def _bucket_from_ivrv(value: Any) -> str:
    try:
        spread = float(value or 0)
    except Exception:
        return "neutral"
    if math.isnan(spread):
        return "neutral"
    if spread <= -5:
        return "cheap"
    if spread <= 15:
        return "neutral"
    return "rich"


def _bucket_from_skew(value: Any) -> str:
    try:
        skew = float(value or 0)
    except Exception:
        return "neutral"
    if math.isnan(skew):
        return "neutral"
    if skew >= 3:
        return "supportive"
    if skew <= -3:
        return "adverse"
    return "neutral"


def _bucket_from_flow(vol_oi_ratio: Any, premium: Any) -> str:
    ratio = _num(vol_oi_ratio)
    premium_value = _num(premium)
    if ratio >= 2.0 or premium_value >= 250000:
        return "strong"
    if ratio >= 0.8 or premium_value >= 50000:
        return "normal"
    return "weak"


def _bucket_from_liquidity(spread_pct: Any) -> str:
    try:
        spread = float(spread_pct or 999)
    except Exception:
        return "fair"
    if math.isnan(spread):
        return "fair"
    if spread <= 6:
        return "tight"
    if spread <= 15:
        return "fair"
    return "wide"


def _signal_horizon_from_dte(value: Any) -> int:
    bucket = _bucket_from_dte(value)
    if bucket == "short":
        return 3
    if bucket == "mid":
        return 7
    return 14


def _preference_label(max_key: str, values: dict[str, float], threshold: float = 0.35) -> str:
    best = values.get(max_key, 0.0)
    other = [v for k, v in values.items() if k != max_key]
    second = max(other) if other else 0.0
    return max_key if best - second >= threshold else "balanced"


def _factor_bucket_payload(source: dict[str, Any]) -> dict[str, str]:
    return {
        "ivrv": str(source.get("factor_bucket_ivrv") or _bucket_from_ivrv(source.get("iv_hv_spread"))),
        "skew": str(source.get("factor_bucket_skew") or _bucket_from_skew(source.get("skew_support"))),
        "flow": str(source.get("factor_bucket_flow") or _bucket_from_flow(source.get("vol_oi_ratio"), source.get("premium"))),
        "liquidity": str(source.get("factor_bucket_liquidity") or _bucket_from_liquidity(source.get("spread_pct"))),
    }


def _new_bucket_stats() -> dict[str, float]:
    return {"count": 0.0, "wins": 0.0, "score_sum": 0.0}


def _environment_keys_from_sample(sample: dict[str, Any], fallback_env: Optional[dict[str, Any]] = None) -> list[str]:
    env = fallback_env or {}
    temp_bucket = sample.get("market_temp_bucket") or env.get("temp_bucket")
    sentiment_bucket = sample.get("market_sentiment_bucket") or env.get("sentiment_bucket")
    valuation_bucket = sample.get("market_valuation_bucket") or env.get("valuation_bucket")
    trend_bucket = sample.get("market_trend_bucket") or env.get("trend_bucket")
    return [
        f"temp_{temp_bucket}" if temp_bucket else "",
        f"sentiment_{sentiment_bucket}" if sentiment_bucket else "",
        f"valuation_{valuation_bucket}" if valuation_bucket else "",
        f"trend_{trend_bucket}" if trend_bucket else "",
    ]


def _update_bucket(bucket: dict[str, float], outcome_score: float, win: bool, sample_weight: float = 1.0) -> None:
    bucket["count"] += sample_weight
    bucket["score_sum"] += outcome_score * sample_weight
    bucket["wins"] += sample_weight if win else 0.0


def _score_bucket_stats(bucket: dict[str, float]) -> float:
    if bucket["count"] <= 0:
        return 0.0
    avg_score = bucket["score_sum"] / max(bucket["count"], 1e-6)
    win_rate = bucket["wins"] / max(bucket["count"], 1e-6)
    sample_boost = min(1.0, bucket["count"] / 8.0)
    score = (win_rate - 0.5) * 5.0 + max(-2.5, min(2.5, avg_score / 12.0))
    return round(score * sample_boost, 2)


def _source_bucket_from_value(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return "other"
    if any(token in text for token in ("unusual", "abnormal")):
        return "unusual"
    if any(token in text for token in ("manual", "hand")):
        return "manual"
    if any(token in text for token in ("pool", "watchlist", "top50")):
        return "pool"
    if "webull" in text or "live" in text:
        return "webull"
    if "scan" in text:
        return "scan"
    return "other"


def _plan_source_bucket(plan: dict[str, Any] | None) -> str:
    if not isinstance(plan, dict):
        return "other"
    return _source_bucket_from_value(
        plan.get("source_bucket")
        or plan.get("source_tag")
        or plan.get("source")
        or plan.get("origin_source")
    )


def _tag_trade_plans_source(plans: list[dict[str, Any]], source_tag: str) -> list[dict[str, Any]]:
    bucket = _source_bucket_from_value(source_tag)
    tagged: list[dict[str, Any]] = []
    for plan in plans or []:
        item = dict(plan)
        item["source_tag"] = str(source_tag or "")
        item["source_bucket"] = bucket
        tagged.append(item)
    return tagged


def _sim_trade_learning_sample(trade: dict[str, Any]) -> dict[str, Any] | None:
    status = str(trade.get("status") or "").lower()
    plan = trade.get("trade_plan", {}) if isinstance(trade.get("trade_plan"), dict) else {}
    opt_type = str(trade.get("option_type") or plan.get("type") or "").upper()
    if opt_type not in {"CALL", "PUT"}:
        return None

    if status == "closed":
        outcome_pct = float(trade.get("realized_pnl_pct", 0) or 0)
        outcome_cash = float(trade.get("realized_pnl", 0) or 0)
        sample_weight = 1.0
        sample_class = "closed"
        win = outcome_pct > 0
    elif status == "open":
        if trade.get("unrealized_pnl_pct") is None:
            return None
        outcome_pct = float(trade.get("unrealized_pnl_pct", 0) or 0)
        outcome_cash = float(trade.get("unrealized_pnl", 0) or 0)
        days_held = int(trade.get("days_held") or 0)
        conviction = min(1.0, abs(outcome_pct) / 40.0)
        maturity = min(1.0, days_held / 7.0)
        sample_weight = round(min(0.45, 0.12 + 0.18 * conviction + 0.15 * maturity), 3)
        sample_class = "open"
        win = outcome_pct >= 5.0
    else:
        return None

    merged = dict(plan)
    merged.setdefault("type", opt_type)
    merged.setdefault("dte", trade.get("dte"))
    merged.setdefault("iv_pct", trade.get("iv_pct"))
    merged.setdefault("risk_reward", trade.get("risk_reward"))
    merged.setdefault("iv_hv_spread", trade.get("iv_hv_spread"))
    merged.setdefault("skew_support", trade.get("skew_support"))
    merged.setdefault("vol_oi_ratio", trade.get("vol_oi_ratio"))
    merged.setdefault("premium", trade.get("premium"))
    merged.setdefault("spread_pct", trade.get("spread_pct"))
    merged.setdefault("source_tag", trade.get("source") or trade.get("origin", {}).get("source"))
    merged.setdefault("source_bucket", _source_bucket_from_value(merged.get("source_tag")))
    merged.setdefault("market_temp_bucket", trade.get("market_temp_bucket"))
    merged.setdefault("market_sentiment_bucket", trade.get("market_sentiment_bucket"))
    merged.setdefault("market_valuation_bucket", trade.get("market_valuation_bucket"))
    merged.setdefault("market_trend_bucket", trade.get("market_trend_bucket"))
    return {
        "status": sample_class,
        "weight": sample_weight,
        "outcome_pct": outcome_pct,
        "outcome_cash": outcome_cash,
        "win": win,
        "plan": merged,
        "symbol": str(trade.get("symbol") or "").upper(),
        "option_type": opt_type,
        "source_bucket": _plan_source_bucket(merged),
        "sample_ts": trade.get("closed_at") or trade.get("updated_at") or trade.get("opened_at"),
        "market_temp_bucket": merged.get("market_temp_bucket"),
        "market_sentiment_bucket": merged.get("market_sentiment_bucket"),
        "market_valuation_bucket": merged.get("market_valuation_bucket"),
        "market_trend_bucket": merged.get("market_trend_bucket"),
    }


def _record_signal_candidates(symbol: str, spot: float, trade_plans: list[dict[str, Any]], tech_bias: str, source: str = "scan") -> None:
    if not symbol or not trade_plans or not spot:
        return
    state = _load_signal_memory()
    now_iso = _now_et_iso()
    market_env = _current_market_environment(refresh=False)
    recent_keys = {
        (
            str(item.get("symbol") or "").upper(),
            str(item.get("expiry") or ""),
            str(item.get("type") or "").upper(),
            _safe(item.get("strike"), 2),
            str(item.get("scan_date") or "")[:13],
        )
        for item in state.get("signals", [])
    }
    for rank, plan in enumerate(trade_plans[:3], start=1):
        key = (
            _normalize_symbol(symbol),
            str(plan.get("expiry") or ""),
            str(plan.get("type") or "").upper(),
            _safe(plan.get("strike"), 2),
            now_iso[:13],
        )
        if key in recent_keys:
            continue
        buckets = _factor_bucket_payload(plan)
        state["signals"].append(
            {
                "id": uuid.uuid4().hex,
                "symbol": _normalize_symbol(symbol),
                "scan_date": now_iso,
                "spot": _safe(spot, 4),
                "type": str(plan.get("type") or "").upper(),
                "expiry": str(plan.get("expiry") or ""),
                "strike": _safe(plan.get("strike"), 2),
                "dte": int(plan.get("dte") or 0),
                "rank": rank,
                "tech_bias": tech_bias,
                "status": "open",
                "horizon_days": _signal_horizon_from_dte(plan.get("dte")),
                "source": source,
                "source_bucket": _source_bucket_from_value(source),
                "iv_pct": _safe(plan.get("iv_pct")),
                "iv_hv_spread": _safe(plan.get("iv_hv_spread")),
                "term_iv_spread": _safe(plan.get("term_iv_spread")),
                "skew_support": _safe(plan.get("skew_support")),
                "vol_oi_ratio": _safe(plan.get("vol_oi_ratio")),
                "premium": _safe(plan.get("premium")),
                "spread_pct": _safe(plan.get("spread_pct")),
                "factor_score": _safe(plan.get("factor_score")),
                "factor_bucket_ivrv": buckets["ivrv"],
                "factor_bucket_skew": buckets["skew"],
                "factor_bucket_flow": buckets["flow"],
                "factor_bucket_liquidity": buckets["liquidity"],
                "market_temp_bucket": market_env.get("temp_bucket"),
                "market_sentiment_bucket": market_env.get("sentiment_bucket"),
                "market_valuation_bucket": market_env.get("valuation_bucket"),
                "market_trend_bucket": market_env.get("trend_bucket"),
                "market_env_label": market_env.get("label"),
            }
        )
    state["signals"] = state.get("signals", [])[-2000:]
    state["updated_at"] = now_iso
    _save_signal_memory(state)


def _refresh_signal_memory() -> dict[str, Any]:
    state = _load_signal_memory()
    signals = state.get("signals", [])
    now = dt.datetime.now(ET)
    changed = False
    resolved_feedback: list[str] = []
    for item in signals:
        if item.get("status") != "open":
            continue
        try:
            opened = dt.datetime.fromisoformat(str(item.get("scan_date")))
            if opened.tzinfo is None:
                opened = ET.localize(opened)
            else:
                opened = opened.astimezone(ET)
        except Exception:
            continue
        if (now - opened).total_seconds() < int(item.get("horizon_days") or 0) * 86400:
            continue
        hist, source = _daily_history(str(item.get("symbol") or ""), max(40, int(item.get("horizon_days") or 0) + 10))
        if hist.empty or "Close" not in hist.columns:
            continue
        current_spot = _safe(hist["Close"].dropna().iloc[-1], 4) if not hist["Close"].dropna().empty else None
        entry_spot = _num(item.get("spot"))
        if current_spot is None or entry_spot <= 0:
            continue
        raw_return = (float(current_spot) / entry_spot - 1.0) * 100.0
        directional = raw_return if str(item.get("type") or "").upper() == "CALL" else -raw_return
        item["status"] = "resolved"
        item["resolved_at"] = _now_et_iso()
        item["current_spot"] = current_spot
        item["underlying_return_pct"] = round(raw_return, 2)
        item["directional_edge_pct"] = round(directional, 2)
        item["success"] = directional > 0.75
        item["history_source"] = source
        resolved_feedback.append(
            f"- {item.get('symbol')}: success={item.get('success')} edge={item.get('directional_edge_pct')} source={item.get('source_bucket')} env={item.get('market_env_label') or 'unknown'}"
        )
        changed = True
    if changed:
        state["updated_at"] = _now_et_iso()
        _save_signal_memory(state)
        if resolved_feedback:
            _append_market_memory("environment_feedback", resolved_feedback[-20:], timestamp=state["updated_at"])
    return state


def _rebuild_learning_state() -> dict[str, Any]:
    now = dt.datetime.now(ET)
    current_env = _current_market_environment(refresh=False)
    state = _refresh_sim_state(_load_sim_state())
    closed = [t for t in state.get("closed", []) if t.get("status") == "closed"]
    open_trades = [t for t in state.get("trades", []) if t.get("status") == "open"]
    signal_state = _refresh_signal_memory()
    resolved_signals = [s for s in signal_state.get("signals", []) if s.get("status") == "resolved"]
    sim_samples = [sample for sample in (_sim_trade_learning_sample(t) for t in closed + open_trades) if sample]
    open_samples = [sample for sample in sim_samples if sample["status"] == "open"]
    open_sample_weight = round(sum(float(sample.get("weight", 0) or 0) for sample in open_samples), 2)
    if not sim_samples and not resolved_signals:
        payload = _default_learning_state()
        payload["updated_at"] = _now_et_iso()
        _save_learning_state(payload)
        return payload

    buckets: dict[str, dict[str, list[float]]] = {
        "CALL": {"pnl": [], "wins": 0, "count": 0},
        "PUT": {"pnl": [], "wins": 0, "count": 0},
        "short": {"pnl": [], "wins": 0, "count": 0},
        "mid": {"pnl": [], "wins": 0, "count": 0},
        "long": {"pnl": [], "wins": 0, "count": 0},
        "iv_low": {"pnl": [], "wins": 0, "count": 0},
        "iv_mid": {"pnl": [], "wins": 0, "count": 0},
        "iv_high": {"pnl": [], "wins": 0, "count": 0},
        "rr_low": {"pnl": [], "wins": 0, "count": 0},
        "rr_mid": {"pnl": [], "wins": 0, "count": 0},
        "rr_high": {"pnl": [], "wins": 0, "count": 0},
    }
    factor_buckets: dict[str, dict[str, float]] = {
        "ivrv_cheap": _new_bucket_stats(),
        "ivrv_neutral": _new_bucket_stats(),
        "ivrv_rich": _new_bucket_stats(),
        "skew_supportive": _new_bucket_stats(),
        "skew_neutral": _new_bucket_stats(),
        "skew_adverse": _new_bucket_stats(),
        "flow_strong": _new_bucket_stats(),
        "flow_normal": _new_bucket_stats(),
        "flow_weak": _new_bucket_stats(),
        "liquidity_tight": _new_bucket_stats(),
        "liquidity_fair": _new_bucket_stats(),
        "liquidity_wide": _new_bucket_stats(),
    }
    source_buckets: dict[str, dict[str, float]] = {
        "scan": _new_bucket_stats(),
        "unusual": _new_bucket_stats(),
        "manual": _new_bucket_stats(),
        "pool": _new_bucket_stats(),
        "webull": _new_bucket_stats(),
        "other": _new_bucket_stats(),
    }
    environment_buckets: dict[str, dict[str, float]] = {
        "temp_hot": _new_bucket_stats(),
        "temp_warm": _new_bucket_stats(),
        "temp_neutral": _new_bucket_stats(),
        "temp_cool": _new_bucket_stats(),
        "temp_cold": _new_bucket_stats(),
        "sentiment_greed": _new_bucket_stats(),
        "sentiment_neutral": _new_bucket_stats(),
        "sentiment_fear": _new_bucket_stats(),
        "valuation_rich": _new_bucket_stats(),
        "valuation_fair": _new_bucket_stats(),
        "valuation_cheap": _new_bucket_stats(),
        "trend_bullish": _new_bucket_stats(),
        "trend_neutral": _new_bucket_stats(),
        "trend_bearish": _new_bucket_stats(),
    }

    symbol_stats: dict[str, dict[str, float]] = {}
    for sample in sim_samples:
        plan = sample["plan"]
        opt_type = sample["option_type"]
        sample_weight = float(sample.get("weight", 1.0) or 1.0) * _recency_multiplier(sample.get("sample_ts"), now=now)
        pnl_pct = float(sample.get("outcome_pct", 0) or 0)
        pnl_cash = float(sample.get("outcome_cash", 0) or 0)
        win = bool(sample.get("win"))
        dte_key = _bucket_from_dte(plan.get("dte"))
        iv_key = f"iv_{_bucket_from_iv(plan.get('iv_pct'))}"
        rr_key = f"rr_{_bucket_from_rr(plan.get('risk_reward'))}"
        for key in [opt_type, dte_key, iv_key, rr_key]:
            if key not in buckets:
                continue
            buckets[key]["count"] += sample_weight
            buckets[key]["pnl"].append(pnl_pct * sample_weight * 4.0)
            buckets[key]["wins"] += sample_weight if win else 0
        symbol = sample["symbol"]
        if symbol:
            symbol_bucket = symbol_stats.setdefault(symbol, {"count": 0.0, "pnl": 0.0, "wins": 0.0})
            symbol_bucket["count"] += sample_weight
            symbol_bucket["pnl"] += pnl_cash if sample["status"] == "closed" else pnl_pct
            symbol_bucket["wins"] += sample_weight if win else 0
        factor_payload = _factor_bucket_payload(plan)
        source_bucket = str(sample.get("source_bucket") or "other")
        _update_bucket(factor_buckets[f"ivrv_{factor_payload['ivrv']}"], pnl_pct, win, sample_weight)
        _update_bucket(factor_buckets[f"skew_{factor_payload['skew']}"], pnl_pct, win, sample_weight)
        _update_bucket(factor_buckets[f"flow_{factor_payload['flow']}"], pnl_pct, win, sample_weight)
        _update_bucket(factor_buckets[f"liquidity_{factor_payload['liquidity']}"], pnl_pct, win, sample_weight)
        if source_bucket in source_buckets:
            _update_bucket(source_buckets[source_bucket], pnl_pct, win, sample_weight)
        for env_key in _environment_keys_from_sample(sample, fallback_env=current_env):
            if env_key in environment_buckets:
                _update_bucket(environment_buckets[env_key], pnl_pct, win, sample_weight)

    for signal in resolved_signals:
        directional = float(signal.get("directional_edge_pct", 0) or 0)
        win = directional > 0.75
        factor_payload = _factor_bucket_payload(signal)
        source_bucket = _source_bucket_from_value(signal.get("source_bucket") or signal.get("source"))
        signal_weight = 0.35 * _recency_multiplier(signal.get("resolved_at") or signal.get("scan_date"), half_life_days=14.0, now=now)
        _update_bucket(factor_buckets[f"ivrv_{factor_payload['ivrv']}"], directional, win, signal_weight)
        _update_bucket(factor_buckets[f"skew_{factor_payload['skew']}"], directional, win, signal_weight)
        _update_bucket(factor_buckets[f"flow_{factor_payload['flow']}"], directional, win, signal_weight)
        _update_bucket(factor_buckets[f"liquidity_{factor_payload['liquidity']}"], directional, win, signal_weight)
        if source_bucket in source_buckets:
            _update_bucket(source_buckets[source_bucket], directional, win, signal_weight)
        for env_key in _environment_keys_from_sample(signal, fallback_env=current_env):
            if env_key in environment_buckets:
                _update_bucket(environment_buckets[env_key], directional, win, signal_weight)

    def _score_bucket(key: str) -> float:
        bucket = buckets[key]
        if bucket["count"] == 0:
            return 0.0
        avg_pnl = sum(bucket["pnl"]) / max(bucket["count"], 1)
        win_rate = bucket["wins"] / max(bucket["count"], 1)
        sample_boost = min(1.0, bucket["count"] / 8.0)
        score = (win_rate - 0.5) * 5.0 + max(-2.5, min(2.5, avg_pnl / 120.0))
        return round(score * sample_boost, 2)

    weights = {
        "type_call": _score_bucket("CALL"),
        "type_put": _score_bucket("PUT"),
        "dte_short": _score_bucket("short"),
        "dte_mid": _score_bucket("mid"),
        "dte_long": _score_bucket("long"),
        "iv_low": _score_bucket("iv_low"),
        "iv_mid": _score_bucket("iv_mid"),
        "iv_high": _score_bucket("iv_high"),
        "rr_low": _score_bucket("rr_low"),
        "rr_mid": _score_bucket("rr_mid"),
        "rr_high": _score_bucket("rr_high"),
    }
    factor_weights = {key: _score_bucket_stats(bucket) for key, bucket in factor_buckets.items()}
    source_weights = {key: _score_bucket_stats(bucket) for key, bucket in source_buckets.items()}
    environment_weights = {key: _score_bucket_stats(bucket) for key, bucket in environment_buckets.items()}

    direction_pref = _preference_label("call", {"call": weights["type_call"], "put": weights["type_put"]})
    dte_pref = _preference_label("short", {"short": weights["dte_short"], "mid": weights["dte_mid"], "long": weights["dte_long"]})
    iv_pref = _preference_label("low", {"low": weights["iv_low"], "mid": weights["iv_mid"], "high": weights["iv_high"]})
    rr_pref = _preference_label("mid", {"low": weights["rr_low"], "mid": weights["rr_mid"], "high": weights["rr_high"]})
    effective_samples = len(closed) + open_sample_weight * 0.6 + len(resolved_signals) * 0.35
    confidence = round(min(0.95, 0.18 + effective_samples / 24.0), 2)

    top_symbols = sorted(
        [
            {
                "symbol": symbol,
                "count": round(values["count"], 2),
                "pnl": round(values["pnl"], 2),
                "win_rate": round(values["wins"] / max(values["count"], 1) * 100, 2),
            }
            for symbol, values in symbol_stats.items()
        ],
        key=lambda item: (item["pnl"], item["win_rate"]),
        reverse=True,
    )[:5]

    signal_win_rate = round(sum(1 for s in resolved_signals if s.get("success")) / len(resolved_signals) * 100, 2) if resolved_signals else None
    open_positive = sum(1 for t in open_trades if float(t.get("unrealized_pnl", 0) or 0) > 0)
    open_negative = sum(1 for t in open_trades if float(t.get("unrealized_pnl", 0) or 0) < 0)
    open_avg_return = round(
        sum(float(t.get("unrealized_pnl_pct", 0) or 0) for t in open_trades if t.get("unrealized_pnl_pct") is not None) /
        max(1, sum(1 for t in open_trades if t.get("unrealized_pnl_pct") is not None)),
        2,
    ) if open_trades else 0.0
    recent_samples_7d = [sample for sample in sim_samples if _within_days(sample.get("sample_ts"), 7, now=now)]
    recent_samples_30d = [sample for sample in sim_samples if _within_days(sample.get("sample_ts"), 30, now=now)]
    recent_signals_7d = [signal for signal in resolved_signals if _within_days(signal.get("resolved_at") or signal.get("scan_date"), 7, now=now)]
    recent_signals_30d = [signal for signal in resolved_signals if _within_days(signal.get("resolved_at") or signal.get("scan_date"), 30, now=now)]

    def _recent_direction(samples: list[dict[str, Any]]) -> str:
        if not samples:
            return "balanced"
        call_score = sum(float(item.get("weight", 0) or 0) for item in samples if item.get("option_type") == "CALL" and item.get("win"))
        put_score = sum(float(item.get("weight", 0) or 0) for item in samples if item.get("option_type") == "PUT" and item.get("win"))
        if call_score - put_score >= 0.5:
            return "call"
        if put_score - call_score >= 0.5:
            return "put"
        return "balanced"

    summary = (
        f"系统当前更偏 {'CALL' if direction_pref == 'call' else 'PUT' if direction_pref == 'put' else '双向'}，"
        f"DTE 偏好 {dte_pref}，IV 偏好 {iv_pref}，风险收益偏好 {rr_pref}，"
        f"真实平仓样本 {len(closed)} 笔，开放模拟持仓 {len(open_trades)} 笔（有效样本权重 {open_sample_weight}），"
        f"自主跟踪信号 {len(resolved_signals)} 条，学习置信度 {confidence}。"
    )
    payload = {
        "updated_at": _now_et_iso(),
        "closed_count": len(closed),
        "open_position_count": len(open_trades),
        "open_position_sample_weight": open_sample_weight,
        "resolved_signal_count": len(resolved_signals),
        "confidence": confidence,
        "summary": summary,
        "preferences": {
            "direction": direction_pref,
            "dte": dte_pref,
            "iv": iv_pref,
            "rr": rr_pref,
        },
        "weights": weights,
        "factor_weights": factor_weights,
        "source_weights": source_weights,
        "stats": {
            "top_symbols": top_symbols,
            "call_win_rate": round(buckets["CALL"]["wins"] / max(buckets["CALL"]["count"], 1) * 100, 2) if buckets["CALL"]["count"] else None,
            "put_win_rate": round(buckets["PUT"]["wins"] / max(buckets["PUT"]["count"], 1) * 100, 2) if buckets["PUT"]["count"] else None,
        },
        "sim_stats": {
            "closed_count": len(closed),
            "open_count": len(open_trades),
            "open_positive_count": open_positive,
            "open_negative_count": open_negative,
            "open_avg_return_pct": open_avg_return,
            "open_sample_weight": open_sample_weight,
        },
        "signal_stats": {
            "resolved_count": len(resolved_signals),
            "signal_win_rate": signal_win_rate,
        },
        "recent_stats": {
            "samples_7d": len(recent_samples_7d),
            "samples_30d": len(recent_samples_30d),
            "resolved_7d": len(recent_signals_7d),
            "resolved_30d": len(recent_signals_30d),
            "direction_7d": _recent_direction(recent_samples_7d),
            "direction_30d": _recent_direction(recent_samples_30d),
        },
        "source_stats": {
            key: {
                "sample_weight": round(bucket["count"], 2),
                "win_rate": round(bucket["wins"] / max(bucket["count"], 1e-6) * 100, 2) if bucket["count"] else None,
                "score": source_weights.get(key, 0.0),
            }
            for key, bucket in source_buckets.items()
            if bucket["count"] > 0
        },
        "environment_weights": environment_weights,
        "environment_stats": {
            key: {
                "sample_weight": round(bucket["count"], 2),
                "win_rate": round(bucket["wins"] / max(bucket["count"], 1e-6) * 100, 2) if bucket["count"] else None,
                "score": environment_weights.get(key, 0.0),
            }
            for key, bucket in environment_buckets.items()
            if bucket["count"] > 0
        },
    }
    payload["knowledge_bias"] = _knowledge_bias_from_learning(payload)
    payload["knowledge_summary"] = payload["knowledge_bias"].get("summary") or "知识库暂未形成稳定规则"
    payload["knowledge_files"] = _knowledge_files_payload()
    _save_learning_state(payload)
    _write_learning_knowledge_wiki(payload, resolved_signals)
    return payload


def _adaptive_learning_profile() -> dict[str, Any]:
    try:
        return _rebuild_learning_state()
    except Exception as exc:
        logger.warning("adaptive learning rebuild failed: %s", exc)
        return _load_learning_state()


def _sim_learning_profile() -> dict[str, Any]:
    state = _load_sim_state()
    closed = [t for t in state.get("closed", []) if t.get("status") == "closed"]
    if not closed:
        return {}

    buckets: dict[str, dict[str, list[float]]] = {
        "CALL": {"wins": [], "losses": []},
        "PUT": {"wins": [], "losses": []},
        "short": {"wins": [], "losses": []},
        "mid": {"wins": [], "losses": []},
        "long": {"wins": [], "losses": []},
    }
    for trade in closed:
        pnl = float(trade.get("realized_pnl", 0) or 0)
        win = pnl > 0
        opt_type = str(trade.get("option_type") or "").upper()
        if opt_type in buckets:
            buckets[opt_type]["wins" if win else "losses"].append(pnl)
        dte = trade.get("trade_plan", {}).get("dte") if isinstance(trade.get("trade_plan"), dict) else trade.get("dte")
        bucket = _bucket_from_dte(dte)
        buckets[bucket]["wins" if win else "losses"].append(pnl)

    def _rate(key: str) -> Optional[float]:
        wins = buckets[key]["wins"]
        losses = buckets[key]["losses"]
        total = len(wins) + len(losses)
        if total < 3:
            return None
        return round(len(wins) / total * 100, 2)

    return {
        "call_win_rate": _rate("CALL"),
        "put_win_rate": _rate("PUT"),
        "short_win_rate": _rate("short"),
        "mid_win_rate": _rate("mid"),
        "long_win_rate": _rate("long"),
        "closed_count": len(closed),
    }


def _sim_learning_bonus(plan: dict[str, Any]) -> float:
    profile = _sim_learning_profile()
    if not profile or not plan:
        return 0.0
    bonus = 0.0
    plan_type = str(plan.get("type") or "").upper()
    dte = int(plan.get("dte") or 0)
    bucket = "short" if dte <= 14 else "mid" if dte <= 35 else "long"

    type_rate = profile.get("call_win_rate") if plan_type == "CALL" else profile.get("put_win_rate") if plan_type == "PUT" else None
    dte_rate = profile.get(f"{bucket}_win_rate")

    if type_rate is not None:
        bonus += max(-3.0, min(3.0, (float(type_rate) - 50.0) * 0.08))
    if dte_rate is not None:
        bonus += max(-2.0, min(2.0, (float(dte_rate) - 50.0) * 0.06))
    return round(bonus, 2)


def _adaptive_learning_bonus(plan: dict[str, Any]) -> float:
    profile = _adaptive_learning_profile()
    if not profile or not plan:
        return 0.0
    weights = profile.get("weights", {}) or {}
    factor_weights = profile.get("factor_weights", {}) or {}
    source_weights = profile.get("source_weights", {}) or {}
    environment_weights = profile.get("environment_weights", {}) or {}
    knowledge_bias = profile.get("knowledge_bias", {}) or {}
    current_env = _current_market_environment(refresh=False)
    bonus = 0.0
    plan_type = str(plan.get("type") or "").upper()
    dte_bucket = _bucket_from_dte(plan.get("dte"))
    iv_bucket = _bucket_from_iv(plan.get("iv_pct"))
    rr_bucket = _bucket_from_rr(plan.get("risk_reward"))

    if plan_type == "CALL":
        bonus += float(weights.get("type_call", 0) or 0)
    elif plan_type == "PUT":
        bonus += float(weights.get("type_put", 0) or 0)
    bonus += float(weights.get(f"dte_{dte_bucket}", 0) or 0)
    bonus += float(weights.get(f"iv_{iv_bucket}", 0) or 0)
    bonus += float(weights.get(f"rr_{rr_bucket}", 0) or 0)
    factor_payload = _factor_bucket_payload(plan)
    bonus += float(factor_weights.get(f"ivrv_{factor_payload['ivrv']}", 0) or 0)
    bonus += float(factor_weights.get(f"skew_{factor_payload['skew']}", 0) or 0)
    bonus += float(factor_weights.get(f"flow_{factor_payload['flow']}", 0) or 0)
    bonus += float(factor_weights.get(f"liquidity_{factor_payload['liquidity']}", 0) or 0)
    source_bucket = _plan_source_bucket(plan)
    bonus += float(source_weights.get(source_bucket, 0) or 0)
    for env_key in (
        f"temp_{current_env.get('temp_bucket')}",
        f"sentiment_{current_env.get('sentiment_bucket')}",
        f"valuation_{current_env.get('valuation_bucket')}",
        f"trend_{current_env.get('trend_bucket')}",
    ):
        bonus += float(environment_weights.get(env_key, 0) or 0) * 0.12
    confidence_gate = _num(knowledge_bias.get("confidence_gate"))
    if confidence_gate >= 0.55:
        recent_7d = str(knowledge_bias.get("recent_direction_7d") or "balanced")
        recent_30d = str(knowledge_bias.get("recent_direction_30d") or "balanced")
        if recent_7d in {"call", "put"}:
            if recent_7d == "call" and plan_type == "CALL":
                bonus += 0.65
            elif recent_7d == "put" and plan_type == "PUT":
                bonus += 0.65
            elif recent_7d == "call" and plan_type == "PUT":
                bonus -= 0.45
            elif recent_7d == "put" and plan_type == "CALL":
                bonus -= 0.45
        elif recent_30d in {"call", "put"}:
            if recent_30d == "call" and plan_type == "CALL":
                bonus += 0.35
            elif recent_30d == "put" and plan_type == "PUT":
                bonus += 0.35
            elif recent_30d == "call" and plan_type == "PUT":
                bonus -= 0.25
            elif recent_30d == "put" and plan_type == "CALL":
                bonus -= 0.25
        preferred_direction = str(knowledge_bias.get("preferred_direction") or "balanced")
        if preferred_direction == "call" and plan_type == "CALL":
            bonus += 0.4
        elif preferred_direction == "put" and plan_type == "PUT":
            bonus += 0.4
        preferred_dte = str(knowledge_bias.get("preferred_dte") or "balanced")
        if preferred_dte == dte_bucket:
            bonus += 0.25
        preferred_iv = str(knowledge_bias.get("preferred_iv") or "balanced")
        if preferred_iv == iv_bucket:
            bonus += 0.2
        weak_sources = {str(item.get("key")) for item in (knowledge_bias.get("weak_sources") or [])}
        favored_sources = {str(item.get("key")) for item in (knowledge_bias.get("favored_sources") or [])}
        if source_bucket in favored_sources:
            bonus += 0.25
        if source_bucket in weak_sources:
            bonus -= 0.35
    return round(max(-6.0, min(6.0, bonus)), 2)


def _refresh_sim_trade(trade: dict[str, Any]) -> dict[str, Any]:
    if trade.get("status") != "open":
        return trade
    symbol = str(trade.get("symbol") or "").upper()
    expiry = str(trade.get("expiry") or "")
    strike = float(trade.get("strike") or 0)
    opt_type = str(trade.get("option_type") or "CALL").upper()
    qty = int(trade.get("qty") or 1)
    entry_price = float(trade.get("entry_price") or 0)
    if not symbol or not expiry or not strike or entry_price <= 0:
        return trade

    underlying = _get_underlying_mark(symbol)
    option = _get_option_mark(symbol, expiry, strike, opt_type)
    current = option.get("mark")
    if current is None:
        current = trade.get("last_mark")
    if current is None:
        return trade

    pnl = (float(current) - entry_price) * qty * SIM_TRADE_MULTIPLIER
    pnl_pct = ((float(current) - entry_price) / entry_price * 100) if entry_price > 0 else None
    snapshot = {
        "ts": _now_et_iso(),
        "mark": _safe(current, 4),
        "underlying": underlying.get("price"),
        "bid": option.get("bid"),
        "ask": option.get("ask"),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "source": option.get("source"),
    }
    history = trade.get("history", [])
    history.append(snapshot)
    trade["history"] = history[-120:]
    trade["last_mark"] = _safe(current, 4)
    trade["last_underlying"] = underlying.get("price")
    trade["unrealized_pnl"] = round(pnl, 2)
    trade["unrealized_pnl_pct"] = round(pnl_pct, 2) if pnl_pct is not None else None
    trade["updated_at"] = snapshot["ts"]
    trade["current_bid"] = option.get("bid")
    trade["current_ask"] = option.get("ask")
    trade["current_source"] = option.get("source")
    trade["days_held"] = max(0, (dt.datetime.fromisoformat(snapshot["ts"]) - dt.datetime.fromisoformat(str(trade.get("opened_at")))).days) if trade.get("opened_at") else trade.get("days_held", 0)
    peak = max((float(x.get("pnl", 0) or 0) for x in trade["history"]), default=pnl)
    trough = min((float(x.get("pnl", 0) or 0) for x in trade["history"]), default=pnl)
    trade["best_pnl"] = round(peak, 2)
    trade["worst_pnl"] = round(trough, 2)
    trade["underlying_snapshot"] = underlying
    return trade


def _refresh_sim_state(state: dict[str, Any]) -> dict[str, Any]:
    trades = state.get("trades", [])
    changed = False
    for idx, trade in enumerate(trades):
        if trade.get("status") != "open":
            continue
        refreshed = _refresh_sim_trade(trade)
        if refreshed is not trade:
            trades[idx] = refreshed
            changed = True
    state["trades"] = trades
    state["updated_at"] = _now_et_iso()
    if changed:
        _save_sim_state(state)
    else:
        _save_sim_state(state)
    return state


def _build_sim_trade(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = _normalize_symbol(payload.get("symbol", ""))
    expiry = str(payload.get("expiry") or "").strip()
    strike = payload.get("strike")
    opt_type = str(payload.get("option_type") or payload.get("type") or "CALL").upper()
    qty = max(1, int(payload.get("qty", 1) or 1))
    if not symbol or not expiry or strike is None:
        raise ValueError("symbol, expiry, strike required")
    strike = float(strike)
    snapshot = _get_option_mark(symbol, expiry, strike, opt_type)
    entry_price = payload.get("entry_price")
    if entry_price is None or float(entry_price) <= 0:
        entry_price = snapshot.get("mark")
    if entry_price is None or float(entry_price) <= 0:
        raise ValueError("无法获取当前期权价格")
    underlying = _get_underlying_mark(symbol)
    opened_at = _now_et_iso()
    market_env = _current_market_environment(refresh=False)
    trade_plan = payload.get("trade_plan") or {}
    trade = {
        "id": uuid.uuid4().hex,
        "symbol": symbol,
        "name": payload.get("name") or payload.get("company_name") or symbol,
        "contract": payload.get("contract") or _lb_option_symbol(symbol, expiry, strike, opt_type),
        "option_type": opt_type,
        "direction": "bullish" if opt_type == "CALL" else "bearish",
        "expiry": expiry,
        "strike": round(strike, 2),
        "qty": qty,
        "entry_price": round(float(entry_price), 4),
        "entry_bid": snapshot.get("bid"),
        "entry_ask": snapshot.get("ask"),
        "entry_source": snapshot.get("source"),
        "opened_at": opened_at,
        "updated_at": opened_at,
        "status": "open",
        "source": payload.get("source") or "manual",
        "notes": payload.get("notes", ""),
        "market_temp_bucket": trade_plan.get("market_temp_bucket") or payload.get("market_temp_bucket") or market_env.get("temp_bucket"),
        "market_sentiment_bucket": trade_plan.get("market_sentiment_bucket") or payload.get("market_sentiment_bucket") or market_env.get("sentiment_bucket"),
        "market_valuation_bucket": trade_plan.get("market_valuation_bucket") or payload.get("market_valuation_bucket") or market_env.get("valuation_bucket"),
        "market_trend_bucket": trade_plan.get("market_trend_bucket") or payload.get("market_trend_bucket") or market_env.get("trend_bucket"),
        "market_env_label": trade_plan.get("market_env_label") or payload.get("market_env_label") or market_env.get("label"),
        "underlying_entry": underlying.get("price"),
        "last_underlying": underlying.get("price"),
        "last_mark": round(float(entry_price), 4),
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "history": [
            {
                "ts": opened_at,
                "mark": round(float(entry_price), 4),
                "underlying": underlying.get("price"),
                "bid": snapshot.get("bid"),
                "ask": snapshot.get("ask"),
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "source": snapshot.get("source"),
            }
        ],
        "trade_plan": trade_plan,
        "origin": payload.get("origin") or {},
    }
    return trade


def _close_sim_trade(state: dict[str, Any], trade_id: str, close_price: float | None = None, note: str = "") -> dict[str, Any]:
    trades = state.get("trades", [])
    for idx, trade in enumerate(trades):
        if trade.get("id") != trade_id:
            continue
        if trade.get("status") != "open":
            return trade
        symbol = str(trade.get("symbol") or "")
        expiry = str(trade.get("expiry") or "")
        strike = float(trade.get("strike") or 0)
        opt_type = str(trade.get("option_type") or "CALL").upper()
        if close_price is None:
            snapshot = _get_option_mark(symbol, expiry, strike, opt_type)
            close_price = snapshot.get("mark")
        if close_price is None:
            raise ValueError("无法获取平仓价格")
        entry_price = float(trade.get("entry_price") or 0)
        qty = int(trade.get("qty") or 1)
        realized = (float(close_price) - entry_price) * qty * SIM_TRADE_MULTIPLIER
        trade["status"] = "closed"
        trade["close_price"] = round(float(close_price), 4)
        trade["closed_at"] = _now_et_iso()
        trade["realized_pnl"] = round(realized, 2)
        trade["realized_pnl_pct"] = round((float(close_price) - entry_price) / entry_price * 100, 2) if entry_price > 0 else None
        trade["close_note"] = note
        trade["updated_at"] = trade["closed_at"]
        state["trades"][idx] = trade
        closed = state.setdefault("closed", [])
        closed.append(dict(trade))
        state["closed"] = closed[-500:]
        return trade
    raise ValueError("trade not found")


def _default_webull_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "auto_execute_on_scan": False,
        "app_key": "",
        "app_secret": "",
        "access_token": "",
        "account_id": "",
        "region_id": "us",
        "api_endpoint": "api.webull.com",
        "max_capital_pct": 0.25,
        "max_open_positions": 2,
        "strategy_notes": "",
        "last_error": "",
        "updated_at": None,
    }


def _load_webull_config() -> dict[str, Any]:
    if not WEBULL_LIVE_CONFIG.exists():
        return _default_webull_config()
    try:
        payload = json.loads(WEBULL_LIVE_CONFIG.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return _default_webull_config()
        cfg = _default_webull_config()
        cfg.update(payload)
        return cfg
    except Exception as exc:
        logger.warning("webull config load failed: %s", exc)
        return _default_webull_config()


def _save_webull_config(config: dict[str, Any]) -> dict[str, Any]:
    payload = _default_webull_config()
    payload.update(config or {})
    payload["updated_at"] = _now_et_iso()
    WEBULL_LIVE_CONFIG.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_iso_timestamp), encoding="utf-8")
    return payload


def _load_webull_state() -> dict[str, Any]:
    if not WEBULL_LIVE_STATE.exists():
        return {"updated_at": None, "executions": [], "last_status": None}
    try:
        payload = json.loads(WEBULL_LIVE_STATE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"updated_at": None, "executions": [], "last_status": None}
        payload.setdefault("executions", [])
        payload.setdefault("updated_at", None)
        payload.setdefault("last_status", None)
        return payload
    except Exception as exc:
        logger.warning("webull state load failed: %s", exc)
        return {"updated_at": None, "executions": [], "last_status": None}


def _save_webull_state(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state or {})
    state["updated_at"] = _now_et_iso()
    state.setdefault("executions", [])
    state.setdefault("last_status", None)
    WEBULL_LIVE_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=_iso_timestamp), encoding="utf-8")
    return state


def _webull_recent_execution_summary(state: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    records = state.get("executions", []) if isinstance(state, dict) else []
    if not isinstance(records, list):
        return []
    out: list[dict[str, Any]] = []
    for item in reversed(records[-limit:]):
        if not isinstance(item, dict):
            continue
        selected = item.get("selected") if isinstance(item.get("selected"), dict) else {}
        out.append(
            {
                "ts": item.get("ts"),
                "symbol": item.get("symbol") or selected.get("symbol"),
                "contract": item.get("contract") or selected.get("contract"),
                "qty": item.get("qty") or selected.get("qty"),
                "limit_price": item.get("limit_price"),
                "status": item.get("status") or ("error" if item.get("error") else "submitted"),
                "error": item.get("error") or "",
                "source": item.get("source") or selected.get("source") or "",
            }
        )
    return out


def _merge_webull_config(base: dict[str, Any] | None, override: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = _default_webull_config()
    merged.update(base or {})
    if override:
        for key, value in override.items():
            if key in {"enabled", "auto_execute_on_scan"}:
                merged[key] = bool(value)
            elif key in {"max_capital_pct"}:
                try:
                    merged[key] = max(0.0, min(1.0, float(value)))
                except Exception:
                    pass
            elif key in {"max_open_positions"}:
                try:
                    merged[key] = max(1, int(value))
                except Exception:
                    pass
            elif key in {"app_key", "app_secret", "access_token", "account_id", "region_id", "api_endpoint", "strategy_notes"}:
                text = str(value or "").strip()
                if text:
                    merged[key] = text
            elif key == "last_error":
                merged[key] = str(value or "")
    merged["region_id"] = str(merged.get("region_id") or "us").strip().lower()
    merged["api_endpoint"] = str(merged.get("api_endpoint") or "api.webull.com").strip().rstrip("/")
    merged["max_capital_pct"] = max(0.0, min(1.0, float(merged.get("max_capital_pct") or 0.25)))
    merged["max_open_positions"] = max(1, int(merged.get("max_open_positions") or 2))
    merged["strategy_notes"] = str(merged.get("strategy_notes") or "").strip()
    return merged


def _mask_webull_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _merge_webull_config(config)
    return {
        "enabled": bool(cfg.get("enabled")),
        "auto_execute_on_scan": bool(cfg.get("auto_execute_on_scan")),
        "app_key": cfg.get("app_key") or "",
        "account_id": cfg.get("account_id") or "",
        "region_id": cfg.get("region_id") or "us",
        "api_endpoint": cfg.get("api_endpoint") or "api.webull.com",
        "max_capital_pct": cfg.get("max_capital_pct"),
        "max_open_positions": cfg.get("max_open_positions"),
        "strategy_notes": cfg.get("strategy_notes") or "",
        "has_app_secret": bool(cfg.get("app_secret")),
        "has_access_token": bool(cfg.get("access_token")),
        "last_error": cfg.get("last_error") or "",
        "updated_at": cfg.get("updated_at"),
    }


def _webull_client(config: dict[str, Any]):
    if WebullApiClient is None or WebullTradeClient is None:
        raise RuntimeError("webull sdk unavailable")
    cfg = _merge_webull_config(config)
    app_key = str(cfg.get("app_key") or "").strip()
    app_secret = str(cfg.get("app_secret") or "").strip()
    if not app_key or not app_secret:
        raise RuntimeError("Webull API key/secret required")

    region_id = cfg.get("region_id") or "us"
    api_client = WebullApiClient(
        app_key,
        app_secret,
        region_id,
        timeout=30,
        connect_timeout=10,
        verify=True,
    )
    api_client.add_endpoint(region_id, cfg.get("api_endpoint") or "api.webull.com")
    access_token = str(cfg.get("access_token") or "").strip()
    if access_token:
        try:
            api_client.set_token(access_token)
        except Exception as exc:
            logger.warning("webull token set failed: %s", exc)
    return api_client, WebullTradeClient(api_client)


def _webull_data_client(config: dict[str, Any]):
    if WebullApiClient is None or WebullDataClient is None:
        raise RuntimeError("webull data sdk unavailable")
    cfg = _merge_webull_config(config)
    app_key = str(cfg.get("app_key") or "").strip()
    app_secret = str(cfg.get("app_secret") or "").strip()
    if not app_key or not app_secret:
        raise RuntimeError("Webull API key/secret required")
    region_id = cfg.get("region_id") or "us"
    api_client = WebullApiClient(
        app_key,
        app_secret,
        region_id,
        timeout=30,
        connect_timeout=10,
        verify=True,
    )
    api_client.add_endpoint(region_id, cfg.get("api_endpoint") or "api.webull.com")
    access_token = str(cfg.get("access_token") or "").strip()
    if access_token:
        try:
            api_client.set_token(access_token)
        except Exception as exc:
            logger.warning("webull token set failed for data client: %s", exc)
    return api_client, WebullDataClient(api_client)


def _webull_response_content(response: Any) -> Any:
    if response is None:
        return None
    for attr in ("get_content", "content", "body"):
        if hasattr(response, attr):
            try:
                value = getattr(response, attr)()
            except TypeError:
                value = getattr(response, attr)
            except Exception:
                value = None
            if value is not None:
                return value
    if isinstance(response, dict):
        return response
    if isinstance(response, (str, bytes, bytearray)):
        return response
    if hasattr(response, "__dict__"):
        data = {k: v for k, v in vars(response).items() if not k.startswith("_")}
        if data:
            return data
    return response


def _webull_response_payload(response: Any) -> dict[str, Any]:
    content = _webull_response_content(response)
    if content is None:
        return {}
    if isinstance(content, dict):
        return content
    if isinstance(content, (bytes, bytearray)):
        try:
            content = content.decode("utf-8", "ignore")
        except Exception:
            return {}
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except Exception:
            return {"raw": content}
    return {"data": content}


def _webull_walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _webull_walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _webull_walk_dicts(item)


def _webull_first_value(payload: Any, keys: list[str]) -> Any:
    for item in _webull_walk_dicts(payload):
        for key in keys:
            if key in item and item.get(key) not in (None, "", []):
                return item.get(key)
    return None


def _webull_market_permission(status_error: str) -> str:
    text = str(status_error or "")
    if "Insufficient permission" in text or "subscribe to stock quotes" in text:
        return "unauthorized"
    if "not found" in text.lower():
        return "not_found"
    if text:
        return "error"
    return "unknown"


def _webull_snapshot_row(payload: Any) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, list) and data:
        row = data[0]
        if isinstance(row, dict):
            return row
    if isinstance(payload, dict):
        for item in _webull_walk_dicts(payload):
            if any(k in item for k in ("last_price", "close", "open", "prev_close", "volume")):
                return item
    return {}


def _webull_price_from_row(row: dict[str, Any]) -> dict[str, Any]:
    price = _webull_num(
        row.get("last_price")
        or row.get("close")
        or row.get("latest_price")
        or row.get("price")
        or row.get("last")
    )
    return {
        "symbol": str(row.get("symbol") or "").strip().upper(),
        "price": _safe(price),
        "open": _safe(_webull_num(row.get("open"))),
        "high": _safe(_webull_num(row.get("high"))),
        "low": _safe(_webull_num(row.get("low"))),
        "prev_close": _safe(_webull_num(row.get("prev_close") or row.get("pre_close"))),
        "volume": int(_webull_num(row.get("volume"), 0)),
        "timestamp": row.get("timestamp") or row.get("time"),
        "source": "webull",
    }


def _webull_bars_to_df(payload: Any) -> pd.DataFrame:
    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    rows = data if isinstance(data, list) else []
    parsed = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        ts = item.get("timestamp") or item.get("time") or item.get("date")
        dt_value = pd.to_datetime(ts, unit="ms", errors="coerce") if isinstance(ts, (int, float)) else pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(dt_value):
            continue
        if getattr(dt_value, "tzinfo", None) is not None:
            dt_value = dt_value.tz_convert(None)
        parsed.append(
            {
                "t": dt_value,
                "Open": _webull_num(item.get("open"), np.nan),
                "High": _webull_num(item.get("high"), np.nan),
                "Low": _webull_num(item.get("low"), np.nan),
                "Close": _webull_num(item.get("close") or item.get("last_price"), np.nan),
                "Volume": _webull_num(item.get("volume"), np.nan),
            }
        )
    if not parsed:
        return _empty_df()
    return pd.DataFrame(parsed).set_index("t").sort_index()


def _webull_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    try:
        text = str(value).replace(",", "").replace("$", "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _webull_count_option_positions(payload: Any) -> tuple[list[dict[str, Any]], int, float]:
    positions: list[dict[str, Any]] = []
    used_capital = 0.0
    seen: set[str] = set()
    for item in _webull_walk_dicts(payload):
        qty = _webull_num(
            item.get("qty")
            or item.get("quantity")
            or item.get("position_qty")
            or item.get("available_qty")
            or item.get("positionQty")
        )
        if not qty or abs(qty) <= 0:
            continue
        instrument_type = str(
            item.get("instrument_type")
            or item.get("security_type")
            or item.get("asset_type")
            or item.get("category")
            or item.get("type")
            or ""
        ).upper()
        symbol = str(
            item.get("symbol")
            or item.get("ticker")
            or item.get("contract_symbol")
            or item.get("name")
            or item.get("security_code")
            or ""
        ).strip()
        text_blob = " ".join(str(v) for v in item.values() if v is not None).upper()
        is_option = "OPTION" in instrument_type or "CALL" in instrument_type or "PUT" in instrument_type or "OPTION" in text_blob
        if not is_option:
            continue
        key = symbol or f"{id(item)}"
        if key in seen:
            continue
        seen.add(key)
        avg_cost = _webull_num(
            item.get("average_cost")
            or item.get("averagePrice")
            or item.get("average_price")
            or item.get("avg_cost")
            or item.get("cost_price")
            or item.get("costPrice")
        )
        market_value = _webull_num(item.get("market_value") or item.get("marketValue") or item.get("value"))
        qty_abs = abs(float(qty))
        if avg_cost is not None and avg_cost > 0:
            used_capital += avg_cost * qty_abs * SIM_TRADE_MULTIPLIER
        elif market_value is not None and market_value > 0:
            used_capital += market_value
        positions.append(
            {
                "symbol": symbol,
                "qty": int(qty_abs),
                "market_value": _safe(market_value),
                "average_cost": _safe(avg_cost),
                "raw": item,
            }
        )
    return positions, len(positions), round(used_capital, 2)


def _webull_balance_summary(payload: Any) -> dict[str, Any]:
    available_cash = _webull_num(_webull_first_value(payload, ["available_cash", "availableCash", "cash_available", "cashAvailable"]))
    cash_balance = _webull_num(_webull_first_value(payload, ["cash_balance", "cashBalance", "cash", "settled_cash", "settledCash"]))
    buying_power = _webull_num(_webull_first_value(payload, ["buying_power", "buyingPower", "buy_power"]))
    total_asset = _webull_num(_webull_first_value(payload, ["total_asset", "totalAsset", "equity", "net_liquidation_value", "netLiquidationValue"]))
    if available_cash is None:
        available_cash = cash_balance if cash_balance is not None else buying_power
    if available_cash is None:
        available_cash = total_asset
    return {
        "available_cash": _safe(available_cash),
        "cash_balance": _safe(cash_balance),
        "buying_power": _safe(buying_power),
        "total_asset": _safe(total_asset),
        "currency": _webull_first_value(payload, ["currency", "cash_currency", "cashCurrency"]) or "USD",
    }


def _webull_select_account_id(config: dict[str, Any], accounts_payload: Any) -> str:
    configured = str(config.get("account_id") or "").strip()
    candidates: list[dict[str, str]] = []
    for item in _webull_walk_dicts(accounts_payload):
        account_id = str(item.get("account_id") or item.get("accountId") or item.get("id") or "").strip()
        account_number = str(item.get("account_number") or item.get("accountNo") or item.get("account_number_mask") or "").strip()
        if account_id:
            candidates.append({"account_id": account_id, "account_number": account_number})
    if configured:
        for item in candidates:
            if configured in {item["account_id"], item["account_number"]}:
                return item["account_id"]
        return configured
    for item in candidates:
        if item["account_id"]:
            return item["account_id"]
    return ""


def _webull_account_record(accounts: list[dict[str, Any]], account_id: str) -> dict[str, Any]:
    target = str(account_id or "").strip()
    for item in accounts:
        if str(item.get("account_id") or "").strip() == target:
            return item
    return {}


def _webull_block_reasons(
    balance: dict[str, Any],
    open_option_positions: int,
    used_capital: float,
    config: dict[str, Any],
    budget_limit: float,
    remaining_budget: float,
) -> list[str]:
    reasons: list[str] = []
    max_open_positions = int(config.get("max_open_positions") or 2)
    available_cash = float(balance.get("available_cash") or 0)
    if open_option_positions >= max_open_positions:
        reasons.append(f"当前期权持仓 {open_option_positions} 已达到上限 {max_open_positions}")
    if available_cash <= 0:
        reasons.append("可用现金不足")
    if budget_limit <= 0:
        reasons.append("风险预算为 0，当前资金占用限制无法开仓")
    elif remaining_budget <= 0:
        reasons.append(f"风险预算已用完，当前已用 {round(used_capital, 2)} / {round(budget_limit, 2)}")
    return reasons


def _webull_build_option_order(plan: dict[str, Any], qty: int, limit_price: float, config: dict[str, Any]) -> dict[str, Any]:
    symbol = _normalize_symbol(plan.get("symbol") or "")
    expiry = str(plan.get("expiry") or "").strip()
    strike = float(plan.get("strike") or 0)
    opt_type = str(plan.get("type") or plan.get("option_type") or "CALL").upper()
    side = str(plan.get("side") or "BUY").upper()
    market = str(config.get("region_id") or "us").upper()
    leg = {
        "market": market,
        "instrument_type": "OPTION",
        "instrument_super_type": "OPTION",
        "symbol": _symbol_root(symbol),
        "underlying_symbol": _symbol_root(symbol),
        "option_type": opt_type,
        "side": side,
        "quantity": qty,
        "qty": qty,
        "strike_price": round(strike, 2),
        "init_exp_date": expiry,
        "expiry": expiry,
    }
    order = {
        "combo_type": "NORMAL",
        "order_type": "LIMIT",
        "quantity": qty,
        "qty": qty,
        "limit_price": round(float(limit_price), 2),
        "price": round(float(limit_price), 2),
        "option_strategy": "SINGLE",
        "side": side,
        "time_in_force": "DAY",
        "tif": "DAY",
        "entrust_type": "QTY",
        "market": market,
        "legs": [leg],
    }
    if expiry:
        order["init_exp_date"] = expiry
    return order


def _webull_client_order_id(symbol: str, expiry: str, opt_type: str) -> str:
    base_symbol = "".join(ch for ch in _symbol_root(symbol).upper() if ch.isalnum())[:6] or "OPT"
    expiry_part = "".join(ch for ch in str(expiry or "") if ch.isdigit())[-8:] or dt.datetime.now(ET).strftime("%Y%m%d")
    side_part = "C" if str(opt_type).upper().startswith("C") else "P"
    rand_part = uuid.uuid4().hex[:10].upper()
    return f"PM{base_symbol}{expiry_part}{side_part}{rand_part}"[:32]


def _webull_notes_bonus(plan: dict[str, Any], notes: str) -> float:
    if not notes:
        return 0.0
    txt = notes.lower()
    bonus = 0.0
    plan_type = str(plan.get("type") or plan.get("option_type") or "").upper()
    dte = int(plan.get("dte") or 0)
    iv_pct = plan.get("iv_pct")
    try:
        iv_pct = float(iv_pct) if iv_pct is not None else None
    except Exception:
        iv_pct = None

    if "call" in txt and plan_type == "CALL":
        bonus += 4
    if "put" in txt and plan_type == "PUT":
        bonus += 4
    if any(k in txt for k in ("短dte", "短周期", "0dte", "1dte", "7dte", "weekly", "周")) and dte <= 14:
        bonus += 3
    if any(k in txt for k in ("中dte", "月", "30dte", "monthly")) and 15 <= dte <= 35:
        bonus += 3
    if any(k in txt for k in ("高iv", "高波动", "波动率高")) and iv_pct is not None and iv_pct >= 45:
        bonus += 3
    if any(k in txt for k in ("低iv", "波动率低", "便宜")) and iv_pct is not None and iv_pct <= 30:
        bonus += 3
    if any(k in txt for k in ("突破", "上破", "trend", "顺势")) and plan_type == "CALL":
        bonus += 2
    if any(k in txt for k in ("跌破", "下破", "breakdown")) and plan_type == "PUT":
        bonus += 2
    if any(k in txt for k in ("流动性", "oi", "成交量", "spread")):
        bonus += min(2.5, float(plan.get("score") or 0) / 20.0)
    return round(bonus, 2)


def _webull_status_hint(last_error: str) -> str:
    err = (last_error or "").upper()
    if "2FA_VERIFY_FAILED" in err or "VERIFY" in err:
        return "需要在 Webull App 里完成一次短信/2FA 验证，然后再点一次测试。"
    if "TOO_MANY_REQUESTS" in err or "429" in err:
        return "触发了限流，先暂停一会儿再测试，避免频繁刷新。"
    if "API KEY/SECRET" in err:
        return "先填写 Webull API Key 和 Secret。"
    if "ACCOUNT_ID" in err:
        return "已经拿到认证，但还需要指定账户 ID。"
    return ""


def _webull_rank_live_plan(plans: list[dict[str, Any]], notes: str = "") -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(plans, list) or not plans:
        return None, []
    ranked = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        base = float(plan.get("score") or 0)
        rr = float(plan.get("risk_reward") or 0)
        liq = float(plan.get("oi") or 0) / 1000.0 + float(plan.get("volume") or 0) / 10000.0
        spread = float(plan.get("spread_pct") or 0)
        iv_pct = plan.get("iv_pct")
        try:
            iv_pct = float(iv_pct) if iv_pct is not None else None
        except Exception:
            iv_pct = None
        iv_bonus = 0.0
        if iv_pct is not None and 20 <= iv_pct <= 55:
            iv_bonus = 2.0
        if iv_pct is not None and iv_pct <= 30:
            iv_bonus += 1.0
        if iv_pct is not None and iv_pct >= 55:
            iv_bonus += 0.5
        total = base + rr * 8 + liq + iv_bonus - min(4.0, spread / 2.0) + _webull_notes_bonus(plan, notes)
        ranked.append({**plan, "_live_score": round(total, 2)})
    ranked.sort(key=lambda x: x.get("_live_score", 0), reverse=True)
    return (ranked[0] if ranked else None), ranked


def _webull_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _merge_webull_config(config)
    out: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled")),
        "auto_execute_on_scan": bool(cfg.get("auto_execute_on_scan")),
        "config": _mask_webull_config(cfg),
        "connected": False,
        "accounts": [],
        "account_id": cfg.get("account_id") or "",
        "account_number": "",
        "balance": {},
        "positions": [],
        "open_option_positions": 0,
        "used_capital": 0.0,
        "budget_limit": 0.0,
        "remaining_budget": 0.0,
        "max_open_positions": int(cfg.get("max_open_positions") or 2),
        "max_capital_pct": float(cfg.get("max_capital_pct") or 0.25),
        "order_blocked": True,
        "block_reasons": [],
        "recent_executions": [],
        "last_error": cfg.get("last_error") or "",
        "hint": _webull_status_hint(cfg.get("last_error") or ""),
    }
    if not cfg.get("app_key") or not cfg.get("app_secret"):
        out["last_error"] = "请先填写 Webull 配置"
        out["hint"] = _webull_status_hint(out["last_error"])
        state = _load_webull_state()
        state["last_status"] = out
        _save_webull_state(state)
        return out
    try:
        api_client, trade_client = _webull_client(cfg)
        accounts_resp = trade_client.account_v2.get_account_list()
        accounts_payload = _webull_response_payload(accounts_resp)
        accounts: list[dict[str, Any]] = []
        for item in _webull_walk_dicts(accounts_payload):
            aid = item.get("account_id") or item.get("accountId") or item.get("id")
            if not aid:
                continue
            record = {
                "account_id": str(aid),
                "account_number": str(item.get("account_number") or item.get("accountNo") or ""),
                "name": item.get("account_name") or item.get("accountName") or item.get("name") or "",
                "type": item.get("account_type") or item.get("accountType") or item.get("type") or "",
                "raw": item,
            }
            if record["account_id"] not in {x["account_id"] for x in accounts}:
                accounts.append(record)
        out["accounts"] = accounts[:10]
        account_id = _webull_select_account_id(cfg, accounts_payload)
        if account_id:
            out["account_id"] = account_id
        account_record = _webull_account_record(accounts, account_id)
        out["account_number"] = str(account_record.get("account_number") or "")

        if not account_id:
            raise RuntimeError("Webull account_id not found")

        balance_resp = trade_client.account_v2.get_account_balance(account_id)
        balance_payload = _webull_response_payload(balance_resp)
        positions_resp = trade_client.account_v2.get_account_position(account_id)
        positions_payload = _webull_response_payload(positions_resp)

        balance = _webull_balance_summary(balance_payload)
        positions, open_option_positions, used_capital = _webull_count_option_positions(positions_payload)
        available_cash = float(balance.get("available_cash") or 0)
        budget_limit = round(max(0.0, available_cash * float(cfg.get("max_capital_pct") or 0.25)), 2)
        remaining_budget = round(max(0.0, budget_limit - used_capital), 2)
        block_reasons = _webull_block_reasons(balance, open_option_positions, used_capital, cfg, budget_limit, remaining_budget)
        out.update(
            {
                "connected": True,
                "balance": balance,
                "positions": positions[:20],
                "open_option_positions": open_option_positions,
                "used_capital": used_capital,
                "budget_limit": budget_limit,
                "remaining_budget": remaining_budget,
                "order_blocked": bool(block_reasons),
                "block_reasons": block_reasons,
                "last_error": "",
                "hint": "",
            }
        )
    except Exception as exc:
        out["last_error"] = str(exc)
        out["hint"] = _webull_status_hint(out["last_error"])
        cfg["last_error"] = str(exc)
        if cfg.get("app_key") or cfg.get("app_secret") or cfg.get("access_token"):
            try:
                _save_webull_config(cfg)
            except Exception:
                pass
    state = _load_webull_state()
    out["recent_executions"] = _webull_recent_execution_summary(state)
    state["last_status"] = out
    _save_webull_state(state)
    return out


def _webull_place_option_order(config: dict[str, Any], plan: dict[str, Any], preview_only: bool = False, prevent_duplicate: bool = False) -> dict[str, Any]:
    cfg = _merge_webull_config(config)
    if not cfg.get("enabled"):
        raise RuntimeError("Webull auto trading is disabled")
    if not plan:
        raise RuntimeError("No option plan supplied")

    if prevent_duplicate:
        state = _load_webull_state()
        recent = list(reversed((state.get("executions", []) or [])[-30:]))
        plan_contract = str(plan.get("contract") or "").strip()
        plan_symbol = str(plan.get("symbol") or "").strip().upper()
        for item in recent:
            if str(item.get("contract") or "").strip() == plan_contract and str(item.get("symbol") or "").strip().upper() == plan_symbol:
                raise RuntimeError(f"duplicate auto trade blocked for {plan_contract or plan_symbol}")

    snapshot = _webull_snapshot(cfg)
    if not snapshot.get("connected"):
        raise RuntimeError(snapshot.get("last_error") or "Webull account unavailable")
    if int(snapshot.get("open_option_positions") or 0) >= int(cfg.get("max_open_positions") or 2):
        raise RuntimeError(f"open option positions reached {cfg.get('max_open_positions')}")

    available_cash = float(snapshot.get("balance", {}).get("available_cash") or 0)
    if available_cash <= 0:
        raise RuntimeError("available cash unavailable")
    budget_limit = float(snapshot.get("budget_limit") or 0)
    remaining_budget = float(snapshot.get("remaining_budget") or 0)
    if budget_limit <= 0 or remaining_budget <= 0:
        raise RuntimeError("capital limit reached")

    limit_price = _webull_num(plan.get("entry") or plan.get("ask") or plan.get("last"))
    if limit_price is None or limit_price <= 0:
        raise RuntimeError("option price unavailable")
    requested_qty = max(1, int(plan.get("qty") or 1))
    contract_cost = float(limit_price) * SIM_TRADE_MULTIPLIER
    max_qty_by_budget = int(remaining_budget // contract_cost)
    qty = min(requested_qty, max_qty_by_budget)
    if qty <= 0:
        raise RuntimeError("not enough budget for one contract")

    order = _webull_build_option_order(plan, qty, float(limit_price), cfg)
    client_combo_order_id = _webull_client_order_id(plan.get("symbol") or "", plan.get("expiry") or "", plan.get("type") or plan.get("option_type") or "")
    order["client_order_id"] = client_combo_order_id
    new_orders = [order]
    api_client, trade_client = _webull_client(cfg)
    account_id = snapshot.get("account_id") or _webull_select_account_id(cfg, {})
    if not account_id:
        raise RuntimeError("Webull account_id not found")

    preview_payload = {}
    try:
        preview_resp = trade_client.order_v2.preview_option(account_id, new_orders, client_combo_order_id=client_combo_order_id)
        preview_payload = _webull_response_payload(preview_resp)
    except Exception as exc:
        preview_payload = {"error": str(exc)}

    result: dict[str, Any] = {
        "account_id": account_id,
        "plan": plan,
        "order": order,
        "preview": preview_payload,
        "snapshot": snapshot,
        "placed": False,
        "auto": bool(cfg.get("enabled")),
        "budget_limit": budget_limit,
        "remaining_budget": remaining_budget,
        "client_combo_order_id": client_combo_order_id,
    }

    if preview_only:
        result["status"] = "preview_only"
        return result

    place_resp = trade_client.order_v2.place_option(account_id, new_orders, client_combo_order_id=client_combo_order_id)
    place_payload = _webull_response_payload(place_resp)
    result["placed"] = True
    result["place"] = place_payload
    result["status"] = "submitted"
    state = _load_webull_state()
    executions = state.setdefault("executions", [])
    executions.append(
        {
            "ts": _now_et_iso(),
            "symbol": plan.get("symbol"),
            "contract": plan.get("contract"),
            "qty": qty,
            "limit_price": round(float(limit_price), 2),
            "client_combo_order_id": client_combo_order_id,
            "type": plan.get("type") or plan.get("option_type"),
            "request": order,
            "preview": preview_payload,
            "place": place_payload,
        }
    )
    state["executions"] = executions[-200:]
    state["last_status"] = snapshot
    _save_webull_state(state)
    return result


def _longbridge_static_info(symbols: list[str]) -> list[Any]:
    ctx = _lb_ctx()
    if ctx is None or not symbols:
        return []
    out = []
    for chunk in _chunked(symbols, 80):
        try:
            out.extend(ctx.static_info(chunk))
        except Exception as exc:
            logger.warning("Longbridge static_info failed: %s", exc)
    return out


def _longbridge_quotes(symbols: list[str]) -> dict[str, Any]:
    ctx = _lb_ctx()
    if ctx is None or not symbols:
        return {}
    try:
        quotes = ctx.quote(symbols)
        return {getattr(q, "symbol", ""): q for q in quotes}
    except Exception as exc:
        logger.warning("Longbridge quote batch failed: %s", exc)
        return {}


def _chat_completion(
    prompt: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    timeout: int = 45,
) -> Optional[str]:
    key = (api_key or "").strip()
    base = (base_url or "").strip().rstrip("/")
    if not key or not base:
        return None
    url = f"{base}/chat/completions"
    payload = {
        "model": (model or "").strip() or "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
                or "You are an options strategist. Return concise Chinese guidance with entry, stop loss, take profit, and rationale.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("ai request failed %s: %s", base, exc)
        return None


def _deepseek_chat(prompt: str, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> Optional[str]:
    return _chat_completion(
        prompt,
        api_key=api_key or DEEPSEEK_API_KEY,
        model=model or DEEPSEEK_MODEL,
        base_url=base_url or DEEPSEEK_BASE_URL,
    )


def _normalize_ai_provider(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    base_url = str(raw.get("base_url", "")).strip().rstrip("/")
    api_key = str(raw.get("api_key", "")).strip()
    model = str(raw.get("model", "")).strip()
    enabled = bool(raw.get("enabled", True))
    system_prompt = str(raw.get("system_prompt", "")).strip()
    if not name or not base_url or not model:
        return None
    return {
        "name": name,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "enabled": enabled,
        "system_prompt": system_prompt,
    }


def _collect_ai_providers(body: dict[str, Any]) -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    raw = body.get("ai_providers")
    if isinstance(raw, list):
        for item in raw:
            provider = _normalize_ai_provider(item)
            if provider and provider.get("enabled"):
                providers.append(provider)

    if providers:
        return providers

    selected = str(body.get("ai_provider", "deepseek")).strip().lower()
    fallback_map = {
        "deepseek": {
            "name": "deepseek",
            "base_url": body.get("deepseek_base_url") or DEEPSEEK_BASE_URL,
            "api_key": body.get("deepseek_api_key") or DEEPSEEK_API_KEY,
            "model": body.get("deepseek_model") or DEEPSEEK_MODEL,
            "enabled": bool(body.get("enable_ai", True)),
        },
        "qwen": {
            "name": "qwen",
            "base_url": body.get("qwen_base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": body.get("qwen_api_key") or "",
            "model": body.get("qwen_model") or "qwen-plus",
            "enabled": bool(body.get("enable_ai", True)),
        },
        "gemini": {
            "name": "gemini",
            "base_url": body.get("gemini_base_url") or "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": body.get("gemini_api_key") or "",
            "model": body.get("gemini_model") or "gemini-2.5-flash",
            "enabled": bool(body.get("enable_ai", True)),
        },
        "ollama": {
            "name": "ollama",
            "base_url": body.get("ollama_base_url") or "http://127.0.0.1:11434/v1",
            "api_key": body.get("ollama_api_key") or "ollama",
            "model": body.get("ollama_model") or "qwen2.5:7b",
            "enabled": bool(body.get("enable_ai", True)),
        },
    }
    provider = fallback_map.get(selected) or fallback_map["deepseek"]
    normalized = _normalize_ai_provider(provider)
    return [normalized] if normalized and normalized.get("enabled") else []


def _run_ai_discussion(
    prompt: str,
    providers: list[dict[str, Any]],
    mode: str = "single",
    primary_provider: str | None = None,
) -> dict[str, Any]:
    mode = (mode or "single").strip().lower()
    if not providers:
        return {"final": None, "rounds": [], "mode": mode, "chosen_provider": None}

    primary_provider = (primary_provider or "").strip().lower()
    ordered = providers[:]
    if primary_provider:
        ordered.sort(key=lambda p: 0 if str(p.get("name", "")).strip().lower() == primary_provider else 1)

    if mode != "council":
        selected = ordered[0]
        text = _chat_completion(
            prompt,
            api_key=selected.get("api_key"),
            model=selected.get("model"),
            base_url=selected.get("base_url"),
            system_prompt=selected.get("system_prompt") or None,
        )
        if not text:
            return {"final": None, "rounds": [], "mode": mode, "chosen_provider": selected.get("name")}
        return {"final": text, "rounds": [{"provider": selected.get("name"), "content": text}], "mode": mode, "chosen_provider": selected.get("name")}

    rounds: list[dict[str, Any]] = []
    council = ordered[:3]
    role_prompts = [
        "你是主分析师，重点看趋势、板块、新闻和入场点，输出偏进攻的结论，但必须给出明确止损和触发条件。",
        "你是风控分析师，重点看流动性、IV、价差、OI、仓位和失效条件，输出偏保守的结论。",
        "你是反对意见分析师，专门找出当前方案最可能出错的地方，并给出更稳的替代方案。",
    ]
    for idx, provider in enumerate(council):
        role_prompt = role_prompts[idx] if idx < len(role_prompts) else role_prompts[-1]
        text = _chat_completion(
            prompt,
            api_key=provider.get("api_key"),
            model=provider.get("model"),
            base_url=provider.get("base_url"),
            system_prompt=(
                (provider.get("system_prompt") or "").strip()
                + ("\n" if provider.get("system_prompt") else "")
                + role_prompt
            ),
        )
        if text:
            rounds.append({"provider": provider.get("name"), "content": text})

    if not rounds:
        return {"final": None, "rounds": [], "mode": mode, "chosen_provider": None}
    if len(rounds) == 1:
        first = rounds[0]
        return {"final": first["content"], "rounds": rounds, "mode": mode, "chosen_provider": first["provider"]}

    synth_provider = council[0]
    discussion = "\n\n".join([f"[{item['provider']}]\n{item['content']}" for item in rounds[:5]])
    synthesis_prompt = (
        "下面是三个AI对同一份期权/股票分析的讨论结果。请综合它们，输出一份最终中文结论。"
        "你必须融合主分析、风控和反对意见，不要只复读最乐观的观点。"
        "最终输出要求分为：1. 最终判断 2. 关键理由 3. 首选标的/合约 4. 入场 5. 止损 6. 止盈 7. 风险边界 8. 备选方案。"
        "如果三个观点冲突，优先保留风控和反对意见里更稳的部分。\n\n"
        f"{discussion}"
    )
    final_text = _chat_completion(
        synthesis_prompt,
        api_key=synth_provider.get("api_key"),
        model=synth_provider.get("model"),
        base_url=synth_provider.get("base_url"),
        system_prompt=synth_provider.get("system_prompt") or None,
    )
    if not final_text:
        final_text = "\n\n".join([f"{item['provider']}: {item['content']}" for item in rounds])
    return {
        "final": final_text,
        "rounds": rounds,
        "mode": mode,
        "chosen_provider": synth_provider.get("name"),
    }


def _atr(hist: pd.DataFrame, period: int = 14) -> Optional[float]:
    if hist.empty or len(hist) < 2:
        return None
    df = hist.copy()
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return _safe(atr)


def _build_trade_plans(symbol: str, hist: pd.DataFrame, contracts: list[dict], spot: float, tech_bias: str, iv_rank: Optional[float]) -> list[dict]:
    if not contracts:
        return []
    recent_high = _safe(hist["High"].tail(10).max()) if not hist.empty else None
    recent_low = _safe(hist["Low"].tail(10).min()) if not hist.empty else None
    atr = _atr(hist, 14) or 0
    plans = []
    iv_rank_val = iv_rank if iv_rank is not None else 50.0

    for c in contracts[:5]:
        entry = c.get("last")
        bid = c.get("bid")
        ask = c.get("ask")
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = round((float(bid) + float(ask)) / 2, 2)
            if ask - bid <= max(0.12, mid * 0.08):
                entry = mid
            else:
                entry = round(float(ask) * 0.98, 2)
        elif entry is None:
            entry = ask if ask is not None else bid

        if entry is None:
            continue

        dte = c.get("dte", 0)
        stop_pct = 0.18 if dte <= 7 else 0.24 if dte <= 21 else 0.30
        tp_pct = 0.45 if dte <= 7 else 0.65 if dte <= 21 else 0.85
        if iv_rank_val >= 70:
            stop_pct += 0.05
        elif iv_rank_val <= 30:
            tp_pct += 0.1

        stop = round(float(entry) * (1 - stop_pct), 2)
        take_profit = round(float(entry) * (1 + tp_pct), 2)
        rr = round((take_profit - entry) / max(entry - stop, 0.01), 2)

        if c["type"] == "CALL":
            trigger = round((recent_high or spot) + max(0.1, atr * 0.15), 2)
            invalidation = round((recent_low or spot) - max(0.1, atr * 0.2), 2)
        else:
            trigger = round((recent_low or spot) - max(0.1, atr * 0.15), 2)
            invalidation = round((recent_high or spot) + max(0.1, atr * 0.2), 2)

        plans.append(
            {
                "symbol": symbol,
                "contract": c.get("name"),
                "type": c.get("type"),
                "expiry": c.get("expiry"),
                "strike": c.get("strike"),
                "lb_symbol": c.get("lb_symbol"),
                "bid": c.get("bid"),
                "ask": c.get("ask"),
                "last": c.get("last"),
                "direction": "bullish" if c["type"] == "CALL" else "bearish",
                "entry": entry,
                "stop_loss": stop,
                "take_profit": take_profit,
                "risk_reward": rr,
                "underlying_trigger": trigger,
                "underlying_invalidation": invalidation,
                "dte": c.get("dte"),
                "score": c.get("score"),
                "iv_pct": c.get("iv_pct"),
                "oi": c.get("oi"),
                "volume": c.get("volume"),
                "spread_pct": c.get("spread_pct"),
                "reason": c.get("scan_reason") or ("high liquidity" if (c.get("oi", 0) or 0) > 0 else "watch spread"),
                "scan_bucket": c.get("scan_bucket"),
                "scan_reason": c.get("scan_reason"),
                "unusual_score": c.get("unusual_score"),
                "vol_oi_ratio": c.get("vol_oi_ratio"),
                "premium": c.get("premium"),
                "factor_score": c.get("factor_score"),
                "iv_hv_spread": c.get("iv_hv_spread"),
                "term_iv_spread": c.get("term_iv_spread"),
                "skew_support": c.get("skew_support"),
                "flow_strength": c.get("flow_strength"),
                "liquidity_score": c.get("liquidity_score"),
                "factor_bucket_ivrv": c.get("factor_bucket_ivrv"),
                "factor_bucket_skew": c.get("factor_bucket_skew"),
                "factor_bucket_flow": c.get("factor_bucket_flow"),
                "factor_bucket_liquidity": c.get("factor_bucket_liquidity"),
            }
        )

    return plans


def _get_hv(hist: pd.DataFrame, days: int = 30) -> Optional[float]:
    if hist.empty or "Close" not in hist:
        return None
    ret = hist["Close"].pct_change().dropna()
    if ret.empty:
        return None
    return _safe(ret.tail(days).std() * (252**0.5) * 100)


def _nearest_atm_iv_map(rows: pd.DataFrame, option_type: str | None = None) -> dict[str, float]:
    if rows.empty or "expiry" not in rows.columns or "iv_pct" not in rows.columns:
        return {}
    work = rows.copy()
    if option_type:
        work = work[work["optionType"].astype(str).str.lower() == option_type.lower()].copy()
    if work.empty:
        return {}
    work["_atm_distance"] = pd.to_numeric(work.get("strike"), errors="coerce")
    work["_atm_distance"] = (work["_atm_distance"] - pd.to_numeric(work.get("spot_ref"), errors="coerce")).abs()
    work = work.dropna(subset=["expiry", "iv_pct", "_atm_distance"])
    if work.empty:
        return {}
    selected = work.sort_values(["expiry", "_atm_distance"]).groupby("expiry", as_index=False).first()
    return {str(row["expiry"]): float(row["iv_pct"]) for _, row in selected.iterrows() if pd.notna(row.get("iv_pct"))}


def _apply_option_factor_model(full: pd.DataFrame, spot: float, hv30: Optional[float], tech_bias: str = "") -> pd.DataFrame:
    if full.empty:
        return full

    scored = full.copy()
    for col in ("strike", "dte", "iv_pct", "volume", "oi", "bid", "ask", "last", "spread_pct", "premium", "vol_oi_ratio"):
        if col not in scored.columns:
            scored[col] = np.nan
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    scored["spot_ref"] = float(spot or 0)
    bid = scored["bid"].fillna(0)
    ask = scored["ask"].fillna(0)
    last = scored["last"].fillna(0)
    mid = pd.Series(
        np.where(
            (bid > 0) & (ask > 0),
            (bid + ask) / 2,
            np.where(last > 0, last, np.where(ask > 0, ask, bid)),
        ),
        index=scored.index,
    )
    spread = pd.Series(np.where((bid > 0) & (ask > 0), ask - bid, np.nan), index=scored.index)
    scored["mid"] = scored.get("mid", mid).fillna(mid).round(4)
    scored["spread_pct"] = scored["spread_pct"].where(
        scored["spread_pct"].notna(),
        np.where((spread.notna()) & (mid > 0), (spread / mid * 100).round(2), np.nan),
    )
    scored["vol_oi_ratio"] = scored["vol_oi_ratio"].where(
        scored["vol_oi_ratio"].notna(),
        np.where(scored["oi"].fillna(0) > 0, scored["volume"].fillna(0) / scored["oi"].replace(0, np.nan), scored["volume"].fillna(0)),
    )
    scored["vol_oi_ratio"] = scored["vol_oi_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    scored["premium"] = scored["premium"].where(
        scored["premium"].notna(),
        (scored["mid"].fillna(0).clip(lower=0) * scored["volume"].fillna(0).clip(lower=0) * 100),
    )

    scored["atm_distance_pct"] = np.where(float(spot or 0) > 0, ((scored["strike"] - float(spot)).abs() / float(spot) * 100), np.nan)
    atm_map = _nearest_atm_iv_map(scored)
    call_atm_map = _nearest_atm_iv_map(scored, "call")
    put_atm_map = _nearest_atm_iv_map(scored, "put")
    if atm_map:
        nearest_expiry = min(atm_map.keys(), key=lambda item: pd.to_datetime(item, errors="coerce"))
        front_atm_iv = atm_map.get(nearest_expiry)
    else:
        front_atm_iv = np.nan

    option_types = scored["optionType"].astype(str).str.upper()
    scored["expiry_atm_iv"] = scored["expiry"].map(atm_map)
    scored["expiry_call_atm_iv"] = scored["expiry"].map(call_atm_map)
    scored["expiry_put_atm_iv"] = scored["expiry"].map(put_atm_map)
    scored["pc_skew_iv"] = scored["expiry_put_atm_iv"] - scored["expiry_call_atm_iv"]
    scored["skew_support"] = np.where(option_types == "CALL", -scored["pc_skew_iv"], scored["pc_skew_iv"])
    scored["iv_hv_spread"] = scored["iv_pct"] - float(hv30) if hv30 is not None else np.nan
    scored["term_iv_spread"] = scored["expiry_atm_iv"] - float(front_atm_iv) if pd.notna(front_atm_iv) else np.nan

    iv_hv = scored["iv_hv_spread"].fillna(0)
    skew_support = scored["skew_support"].fillna(0)
    term_spread = scored["term_iv_spread"].fillna(0)
    spread_pct = scored["spread_pct"].fillna(25).clip(lower=0)
    flow_ratio = scored["vol_oi_ratio"].fillna(0).clip(lower=0)
    premium = scored["premium"].fillna(0).clip(lower=0)
    flow_strength = np.minimum(12, np.log1p(flow_ratio) * 3.5 + np.log10(premium + 1) * 1.6)
    liq_component = np.clip(7.0 - spread_pct * 0.38, -8.0, 7.0)
    ivrv_component = np.where(iv_hv <= -5, 8.0, np.where(iv_hv <= 10, 4.0, np.where(iv_hv <= 25, 0.0, -6.0)))
    skew_component = np.clip(skew_support * 0.65, -7.0, 7.0)
    term_component = np.where(term_spread <= 4, 2.5, np.where(term_spread <= 12, 0.5, -2.5))
    align_component = option_types.map(lambda value: _unusual_direction_bonus(value, tech_bias) * 0.6)
    atm_component = np.clip(5.0 - scored["atm_distance_pct"].fillna(30) * 0.35, -6.0, 5.0)

    scored["factor_score"] = (
        ivrv_component
        + skew_component
        + term_component
        + flow_strength
        + liq_component
        + align_component.fillna(0)
        + atm_component
    ).round(2)
    scored["flow_strength"] = flow_strength.round(2)
    scored["liquidity_score"] = liq_component.round(2)
    scored["factor_bucket_ivrv"] = scored["iv_hv_spread"].map(_bucket_from_ivrv)
    scored["factor_bucket_skew"] = scored["skew_support"].map(_bucket_from_skew)
    scored["factor_bucket_flow"] = [
        _bucket_from_flow(ratio, prem) for ratio, prem in zip(scored["vol_oi_ratio"], scored["premium"])
    ]
    scored["factor_bucket_liquidity"] = scored["spread_pct"].map(_bucket_from_liquidity)

    if "score" in scored.columns:
        scored["score"] = pd.to_numeric(scored["score"], errors="coerce").fillna(0) + scored["factor_score"] * 0.55
    if "unusual_score" in scored.columns:
        scored["unusual_score"] = pd.to_numeric(scored["unusual_score"], errors="coerce").fillna(0) + scored["factor_score"] * 0.8
    return scored


def _compute_vwap(intraday: pd.DataFrame):
    if intraday.empty:
        return None
    df = intraday.copy()
    df["tp"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["vwap"] = (df["tp"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


def _bs_greeks(S, K, T, r, sigma, opt_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {}
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        npd1 = norm.pdf(d1)
        gamma = npd1 / (S * sigma * math.sqrt(T))
        vega = S * npd1 * math.sqrt(T) / 100
        if opt_type == "call":
            delta = norm.cdf(d1)
            theta = (-(S * npd1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (-(S * npd1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        return {
            "delta": _safe(delta, 4),
            "gamma": _safe(gamma, 4),
            "theta": _safe(theta, 4),
            "vega": _safe(vega, 4),
            "rho": _safe(rho, 4),
        }
    except Exception:
        return {}


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=True).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=True).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _tech_signal(spot, sma20, sma50, rsi) -> str:
    if spot > sma20 and spot > sma50 and rsi > 55:
        return "bullish_breakout"
    if spot < sma20 and spot < sma50 and rsi < 45:
        return "bearish_breakdown"
    if rsi > 70:
        return "overbought"
    if rsi < 30:
        return "oversold"
    if spot > sma20:
        return "mild_bullish"
    if spot < sma20:
        return "mild_bearish"
    return "neutral"


def _iv_signal(iv_rank, iv_skew) -> str:
    parts = []
    if iv_rank is not None:
        if iv_rank >= 70:
            parts.append("IV偏高, 适合偏空波动率策略")
        elif iv_rank <= 30:
            parts.append("IV偏低, 适合偏多波动率策略")
        else:
            parts.append("IV中性, 更适合方向性策略")
    if iv_skew is not None:
        if iv_skew > 3:
            parts.append("Put 偏斜明显, 下行保护需求更高")
        elif iv_skew < -2:
            parts.append("Call 偏斜明显, 市场更偏上行定价")
    return " | ".join(parts) if parts else "无明显信号"


def _strategy_text(
    symbol,
    contracts,
    spot,
    tech_bias,
    atm_iv,
    iv_rank,
    trade_plans: Optional[list[dict]] = None,
    library_conclusion: Optional[dict[str, Any]] = None,
) -> dict:
    if not contracts:
        return {}

    top = contracts[0]
    best_plan = (trade_plans or [None])[0] or {}
    requested_direction = None
    conclusion = library_conclusion or {}
    direction = 'bullish' if 'bullish' in tech_bias else 'bearish' if 'bearish' in tech_bias else 'neutral'
    plan_type = str(best_plan.get('type') or top.get('type') or '').upper() or 'CALL'
    tech_type = _tech_direction_from_bias(tech_bias)
    best_contract = best_plan.get('contract') or top.get('name', '')
    best_entry = best_plan.get('entry') if best_plan.get('entry') is not None else top.get('last')
    best_stop = best_plan.get('stop_loss')
    best_tp = best_plan.get('take_profit')
    iv_mode = '高位' if (iv_rank is not None and iv_rank >= 70) else '低位' if (iv_rank is not None and iv_rank <= 30) else '中位'
    dte_value = int(best_plan.get('dte') or top.get('dte') or 0)
    dte_mode = '短DTE' if dte_value <= 7 else '中DTE' if dte_value <= 21 else '长DTE'
    library_summary = conclusion.get('summary') or conclusion.get('direction') or '未形成画像'
    library_direction = conclusion.get('direction', 'neutral')
    library_dte = conclusion.get('dte_style', 'mixed')
    library_iv = conclusion.get('iv_style', 'balanced')
    library_structure = conclusion.get('structure', 'both')
    confidence = conclusion.get('confidence', 0)
    sim_learning = _sim_learning_profile()
    adaptive_learning = _adaptive_learning_profile()
    tech_plan = None
    if tech_type and tech_type != plan_type:
        tech_plan = next(
            (plan for plan in (trade_plans or []) if str(plan.get("type") or "").upper() == tech_type and plan.get("contract") != best_contract),
            None,
        )
    alternate_plan = next((plan for plan in (trade_plans or []) if str(plan.get("type") or "").upper() != plan_type), None)

    lines = [
        f'价格参考: {spot}; 技术偏向: {tech_bias}; ATM IV {atm_iv}%; IV Rank {iv_rank}',
        f'单腿候选: {best_contract} | {top.get("expiry", "")} | Strike {best_plan.get("strike", top.get("strike", ""))} | {plan_type}',
        f'盘口: bid/ask {top.get("bid", "")}/{top.get("ask", "")}; volume {top.get("volume", "")}; OI {top.get("oi", "")}; IV {top.get("iv_pct", "")}%',
        f'经验库: {library_summary} | 方向 {library_direction} | DTE {library_dte} | IV {library_iv} | 结构 {library_structure} | 信心 {confidence}',
    ]
    if sim_learning:
        lines.append(
            f"模拟仓学习: CALL {sim_learning.get('call_win_rate', '—')} / PUT {sim_learning.get('put_win_rate', '—')} / 短DTE {sim_learning.get('short_win_rate', '—')} / 中DTE {sim_learning.get('mid_win_rate', '—')} / 长DTE {sim_learning.get('long_win_rate', '—')}"
        )
    if adaptive_learning:
        lines.append(
            f"自主学习: {adaptive_learning.get('summary', '—')}"
        )
    if tech_plan:
        lines.append(
            f"技术补充: 当前技术面更偏向 {tech_type}，已同步纳入 {tech_plan.get('contract')}，入场 {tech_plan.get('entry', '待确认')}，止损 {tech_plan.get('stop_loss', '待确认')}，止盈 {tech_plan.get('take_profit', '待确认')}"
        )
    elif alternate_plan:
        lines.append(
            f"备选方向: 已保留另一侧 {alternate_plan.get('type')} 合约 {alternate_plan.get('contract')}，用于触发失败后的方向切换。"
        )

    if 'bullish' in tech_bias:
        lines.append('触发: 等待正股突破或关键位确认，避免在横盘里过早买 theta')
    elif 'bearish' in tech_bias:
        lines.append('触发: 等待正股破位确认，避免反弹假信号')
    else:
        lines.append('触发: 方向不明时优先考虑价差和限价单，降低单腿风险')

    lines.append('风控: 优先看流动性和价差，其次看 IV Rank，最后才看单纯方向')

    structure_sections = [
        {
            'title': '结构判断',
            'bullets': [
                f'标的当前偏向: {direction}',
                f'当前波动环境: ATM IV {atm_iv}% / IV Rank {iv_rank}',
                f'经验库画像: {library_summary}',
            ],
        },
        {
            'title': '期权候选',
            'bullets': [
                f'首选合约: {best_contract}',
                f'执行方式: {plan_type} | {dte_mode} | {iv_mode}',
                f'建议进场: {best_entry if best_entry is not None else "待确认"}',
                f'建议止损 / 止盈: {best_stop if best_stop is not None else "待确认"} / {best_tp if best_tp is not None else "待确认"}',
                f'技术补充: {tech_plan.get("contract") if tech_plan else (alternate_plan.get("contract") if alternate_plan else "暂无")}',
            ],
        },
        {
            'title': '风控边界',
            'bullets': [
                f'失效信号: {best_plan.get("underlying_invalidation", "待确认")}',
                f'触发信号: {best_plan.get("underlying_trigger", "待确认")}',
                '关键约束: 只做流动性更好的合约，避免价差过宽',
            ],
        },
    ]

    scenario_cards = [
        {
            'title': '主情景',
            'label': '顺势执行',
            'desc': f'若标的按预期方向完成触发，优先沿 {plan_type} 执行，入场以限价单贴近中间价。',
            'meta': f'目标: {best_tp if best_tp is not None else "待确认"}',
        },
        {
            'title': '备选情景',
            'label': '等待确认',
            'desc': '若价格未突破关键位，维持观察，不在噪音区追单。',
            'meta': f'失效: {best_plan.get("underlying_invalidation", "待确认")}',
        },
        {
            'title': '反向情景',
            'label': '撤退',
            'desc': f'若主方向失效，切换观察 {tech_plan.get("contract") if tech_plan else (alternate_plan.get("contract") if alternate_plan else "另一侧结构")}，而不是继续硬扛原方向。',
            'meta': f'触发: {(tech_plan or alternate_plan or {}).get("underlying_trigger", "待确认")}',
        },
    ]

    risk_checks = [
        f'价差: {best_plan.get("reason", "优先看流动性与价差")}',
        '持有纪律: 计划外不加仓，达到止损立即执行',
        '仓位纪律: 单笔风险不超过账户可承受范围',
    ]

    return {
        'title': f'{symbol} 扫描总览',
        'summary': f'{symbol} 当前更适合 {direction} 侧，优先看 {plan_type}，执行以 {dte_mode} 和 {iv_mode} 为主。',
        'lines': lines,
        'top_contract': top,
        'best_plan': best_plan,
        'direction': direction,
        'iv_mode': iv_mode,
        'dte_mode': dte_mode,
        'structure_sections': structure_sections,
        'scenario_cards': scenario_cards,
        'risk_checks': risk_checks,
        'adaptive_learning': adaptive_learning,
        'tech_plan': tech_plan,
        'alternate_plan': alternate_plan,
    }


def _build_option_candidates(
    symbol: str,
    spot: float,
    min_dte: int,
    max_dte: int,
    opt_filter: str,
    otm_range: float,
    enrich_quotes: bool = True,
    allow_yfinance: bool = True,
    allow_longbridge: bool = True,
) -> pd.DataFrame:
    lb_full = _longbridge_option_candidates(symbol, spot, min_dte, max_dte, opt_filter, otm_range, enrich_quotes=enrich_quotes) if allow_longbridge else _empty_df()
    if not lb_full.empty and (_has_option_activity_data(lb_full) or not allow_yfinance):
        return lb_full

    if not allow_yfinance:
        return lb_full if not lb_full.empty else _empty_df()

    ticker = _yf_ticker(symbol)
    try:
        expiries = ticker.options or []
    except Exception as exc:
        if not lb_full.empty:
            logger.debug("%s yahoo options expiry fetch failed, using longbridge fallback: %s", symbol, exc)
            return lb_full
        logger.warning("%s yahoo options expiry fetch failed: %s", symbol, exc)
        return lb_full if not lb_full.empty else _empty_df()
    today = dt.date.today()
    frames = []

    for exp_str in expiries:
        try:
            exp_date = dt.date.fromisoformat(exp_str)
        except Exception:
            continue
        dte = (exp_date - today).days
        if dte < min_dte or dte > max_dte:
            continue

        try:
            chain = ticker.option_chain(exp_str)
        except Exception as exc:
            level = logger.debug if not lb_full.empty else logger.warning
            level("%s option_chain %s failed: %s", symbol, exp_str, exc)
            continue

        for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
            if opt_filter not in ("both", opt_type):
                continue
            if df.empty:
                continue
            item = df.copy()
            item["optionType"] = opt_type
            item["expiry"] = exp_str
            item["dte"] = dte
            frames.append(item)

    if not frames:
        return _empty_df()

    full = pd.concat(frames, ignore_index=True)
    lo = spot * (1 - otm_range)
    hi = spot * (1 + otm_range)
    full = full[(full["strike"] >= lo) & (full["strike"] <= hi)].copy()
    if full.empty:
        return full

    full.rename(
        columns={
            "impliedVolatility": "iv",
            "openInterest": "oi",
            "lastPrice": "last",
        },
        inplace=True,
    )

    for col in ["bid", "ask", "last", "volume", "oi", "iv"]:
        if col not in full.columns:
            full[col] = np.nan

    full["iv_pct"] = (full["iv"] * 100).round(2)
    full["moneyness"] = ((full["strike"] - spot) / spot * 100).round(2)
    full["spread"] = (full["ask"] - full["bid"]).round(3)
    full["spread_pct"] = np.where(
        full["last"].fillna(0) > 0,
        (full["spread"] / full["last"] * 100).round(2),
        np.nan,
    )
    return full


def _score_options(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return full
    scored = full.copy()
    if "spread_pct" in scored.columns:
        spread_source = scored["spread_pct"]
    else:
        spread_source = pd.Series(np.nan, index=scored.index)
    spread_component = np.where(
        spread_source.notna(),
        (1 / (spread_source.fillna(999) + 1)) * 100,
        10.0,
    )
    scored["score"] = (
        scored["volume"].fillna(0) * 0.3
        + scored["oi"].fillna(0) * 0.3
        + spread_component * 0.15
        + scored["iv_pct"].between(15, 50).astype(int) * 20 * 0.15
        + (1 / (scored["moneyness"].abs().fillna(999) + 1)) * 20
    )
    return scored


def _tech_direction_from_bias(tech_bias: str) -> str:
    tech_bias = str(tech_bias or "").lower()
    if "bullish" in tech_bias or tech_bias == "oversold":
        return "CALL"
    if "bearish" in tech_bias or tech_bias == "overbought":
        return "PUT"
    return ""


def _requested_option_type(opt_filter: str) -> str:
    opt_filter = str(opt_filter or "").lower()
    if opt_filter == "call":
        return "CALL"
    if opt_filter == "put":
        return "PUT"
    return ""


def _should_expand_scan_filter(opt_filter: str, tech_bias: str) -> bool:
    requested_type = _requested_option_type(opt_filter)
    tech_type = _tech_direction_from_bias(tech_bias)
    return bool(requested_type and tech_type and requested_type != tech_type)


def _select_scan_rows(full: pd.DataFrame, top_n: int, opt_filter: str, tech_bias: str) -> pd.DataFrame:
    if full.empty:
        return full

    ranked = full.copy()
    requested_type = _requested_option_type(opt_filter)
    tech_type = _tech_direction_from_bias(tech_bias)
    option_types = ranked["optionType"].astype(str).str.upper()

    ranked["requested_bonus"] = np.where(option_types == requested_type, 7.0, 0.0) if requested_type else 0.0
    ranked["tech_bonus"] = option_types.map(lambda value: _unusual_direction_bonus(value, tech_bias))
    ranked["counter_tech_bonus"] = (
        np.where(option_types == tech_type, 8.0, 0.0) if requested_type and tech_type and requested_type != tech_type else 0.0
    )
    ranked["unusual_support_bonus"] = np.minimum(ranked.get("unusual_score", pd.Series(0.0, index=ranked.index)).fillna(0) * 0.12, 9.0)
    ranked["activity_bonus"] = np.where(
        (ranked.get("vol_oi_ratio", pd.Series(0.0, index=ranked.index)).fillna(0) >= 1.0)
        | (ranked.get("premium", pd.Series(0.0, index=ranked.index)).fillna(0) >= 50000),
        4.0,
        0.0,
    )
    ranked["scan_score"] = (
        ranked["score"].fillna(0)
        + ranked["requested_bonus"].fillna(0)
        + ranked["tech_bonus"].fillna(0)
        + ranked["counter_tech_bonus"].fillna(0)
        + ranked["unusual_support_bonus"].fillna(0)
        + ranked["activity_bonus"].fillna(0)
    ).round(2)
    ranked = ranked.sort_values(["scan_score", "score", "unusual_score"], ascending=False)

    chosen_keys: set[str] = set()
    chosen_rows: list[pd.Series] = []

    def _row_key(row: pd.Series) -> str:
        return f"{row.get('expiry')}|{row.get('strike')}|{row.get('optionType')}"

    def _append_first(mask: pd.Series, bucket: str, reason: str) -> None:
        nonlocal chosen_rows
        subset = ranked.loc[mask]
        if subset.empty:
            return
        for _, row in subset.iterrows():
            key = _row_key(row)
            if key in chosen_keys:
                continue
            item = row.copy()
            item["scan_bucket"] = bucket
            item["scan_reason"] = reason
            chosen_rows.append(item)
            chosen_keys.add(key)
            return

    if requested_type:
        _append_first(option_types == requested_type, "primary", f"按当前筛选方向 {requested_type} 保留主方案")
    if requested_type and tech_type and requested_type != tech_type:
        _append_first(option_types == tech_type, "tech_counter", f"技术面偏向 {tech_type}，补充另一侧方案")

    unusual_mask = (
        (ranked.get("vol_oi_ratio", pd.Series(0.0, index=ranked.index)).fillna(0) >= 1.0)
        | (ranked.get("premium", pd.Series(0.0, index=ranked.index)).fillna(0) >= 50000)
        | (ranked.get("unusual_score", pd.Series(0.0, index=ranked.index)).fillna(0) >= 30)
    )
    _append_first(unusual_mask, "unusual_focus", "异常成交与活跃度较高，纳入扫描补充")

    for _, row in ranked.iterrows():
        if len(chosen_rows) >= max(1, top_n):
            break
        key = _row_key(row)
        if key in chosen_keys:
            continue
        item = row.copy()
        item["scan_bucket"] = item.get("scan_bucket") or "ranked"
        item["scan_reason"] = item.get("scan_reason") or "综合流动性、技术面与异常活跃度排序"
        chosen_rows.append(item)
        chosen_keys.add(key)

    if not chosen_rows:
        return ranked.head(top_n).copy()
    return pd.DataFrame(chosen_rows).head(top_n).copy()


def _enrich_with_longbridge(symbol: str, rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    lb_symbols = [
        _lb_option_symbol(symbol, row["expiry"], row["strike"], row["optionType"])
        for _, row in rows.iterrows()
    ]
    lb_quotes = _longbridge_option_quotes(lb_symbols)
    if not lb_quotes:
        return rows

    enriched = rows.copy()
    last_values = []
    iv_values = []
    oi_values = []
    volume_values = []

    for _, row in enriched.iterrows():
        lb_symbol = _lb_option_symbol(symbol, row["expiry"], row["strike"], row["optionType"])
        q = lb_quotes.get(lb_symbol)
        if q is None:
            last_values.append(row.get("last"))
            iv_values.append(row.get("iv_pct"))
            oi_values.append(row.get("oi"))
            volume_values.append(row.get("volume"))
            continue

        lb_last = getattr(q, "last_done", None)
        lb_iv = getattr(q, "implied_volatility", None)
        lb_oi = getattr(q, "open_interest", None)
        lb_volume = getattr(q, "volume", None)

        last_values.append(_safe(lb_last) if lb_last is not None else row.get("last"))
        iv_values.append(_safe(lb_iv * 100) if lb_iv is not None else row.get("iv_pct"))
        oi_values.append(int(lb_oi) if lb_oi is not None else row.get("oi"))
        volume_values.append(int(lb_volume) if lb_volume is not None else row.get("volume"))

    enriched["last"] = last_values
    enriched["iv_pct"] = iv_values
    enriched["oi"] = oi_values
    enriched["volume"] = volume_values
    return enriched


def _option_contracts(symbol: str, rows: pd.DataFrame) -> list[dict]:
    contracts = []
    for _, row in rows.iterrows():
        contracts.append(
            {
                "name": f"{symbol}{str(row['expiry']).replace('-', '')}{'C' if row['optionType'] == 'call' else 'P'}{int(round(float(row['strike']) * 1000)):08d}",
                "type": row["optionType"].upper(),
                "expiry": row["expiry"],
                "dte": int(row["dte"]),
                "strike": _safe(row["strike"]),
                "lb_symbol": row.get("lb_symbol") if pd.notna(row.get("lb_symbol")) else None,
                "bid": _safe(row.get("bid")),
                "ask": _safe(row.get("ask")),
                "last": _safe(row.get("last")),
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                "oi": int(row["oi"]) if pd.notna(row.get("oi")) else 0,
                "iv_pct": _safe(row["iv_pct"]),
                "moneyness": _safe(row["moneyness"]),
                "spread": _safe(row.get("spread")),
                "spread_pct": _safe(row.get("spread_pct")),
                "delta": _safe(row.get("delta"), 4),
                "gamma": _safe(row.get("gamma"), 4),
                "theta": _safe(row.get("theta"), 4),
                "vega": _safe(row.get("vega"), 4),
                "score": _safe(row.get("score")),
                "unusual_score": _safe(row.get("unusual_score")),
                "vol_oi_ratio": _safe(row.get("vol_oi_ratio")),
                "premium": _safe(row.get("premium")),
                "scan_score": _safe(row.get("scan_score")),
                "scan_bucket": row.get("scan_bucket"),
                "scan_reason": row.get("scan_reason"),
                "factor_score": _safe(row.get("factor_score")),
                "iv_hv_spread": _safe(row.get("iv_hv_spread")),
                "term_iv_spread": _safe(row.get("term_iv_spread")),
                "skew_support": _safe(row.get("skew_support")),
                "flow_strength": _safe(row.get("flow_strength")),
                "liquidity_score": _safe(row.get("liquidity_score")),
                "factor_bucket_ivrv": row.get("factor_bucket_ivrv"),
                "factor_bucket_skew": row.get("factor_bucket_skew"),
                "factor_bucket_flow": row.get("factor_bucket_flow"),
                "factor_bucket_liquidity": row.get("factor_bucket_liquidity"),
            }
        )
    return contracts


def _load_unusual_cache(cache_key: dict[str, Any], max_age_hours: float = 4.0) -> Optional[dict[str, Any]]:
    if not UNUSUAL_OPTIONS_CACHE.exists():
        return None
    try:
        payload = json.loads(UNUSUAL_OPTIONS_CACHE.read_text(encoding="utf-8"))
        fetched_at = _parse_utc_datetime(payload.get("fetched_at"))
        if fetched_at is None:
            return None
        age = (_utc_now() - fetched_at).total_seconds()
        if age > max_age_hours * 3600:
            return None
        if payload.get("cache_key") != cache_key:
            return None
        payload["cached"] = True
        return payload
    except Exception as exc:
        logger.debug("unusual options cache load failed: %s", exc)
        return None


def _load_latest_nonempty_unusual_cache(max_age_hours: float = 24.0) -> Optional[dict[str, Any]]:
    if not UNUSUAL_OPTIONS_CACHE.exists():
        return None
    try:
        payload = json.loads(UNUSUAL_OPTIONS_CACHE.read_text(encoding="utf-8"))
        fetched_at = _parse_utc_datetime(payload.get("fetched_at"))
        if fetched_at is None:
            return None
        age = (_utc_now() - fetched_at).total_seconds()
        if age > max_age_hours * 3600:
            return None
        rows = payload.get("rows") or []
        if not rows:
            return None
        payload["cached"] = True
        payload["stale_fallback"] = True
        return payload
    except Exception as exc:
        logger.debug("latest nonempty unusual cache load failed: %s", exc)
        return None


def _save_unusual_cache(payload: dict[str, Any]) -> None:
    try:
        UNUSUAL_OPTIONS_CACHE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_iso_timestamp),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("unusual options cache save failed: %s", exc)


def _load_top50_cache(cache_key: dict[str, Any], max_age_hours: float = 4.0) -> Optional[dict[str, Any]]:
    if not TOP50_CACHE.exists():
        return None
    try:
        payload = json.loads(TOP50_CACHE.read_text(encoding="utf-8"))
        fetched_at = _parse_utc_datetime(payload.get("fetched_at"))
        if fetched_at is None:
            return None
        age = (_utc_now() - fetched_at).total_seconds()
        if age > max_age_hours * 3600:
            return None
        if payload.get("cache_key") != cache_key:
            return None
        return payload
    except Exception as exc:
        logger.warning("top50 cache load failed: %s", exc)
        return None


def _save_top50_cache(payload: dict[str, Any]) -> None:
    try:
        TOP50_CACHE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_iso_timestamp),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("top50 cache save failed: %s", exc)


def _load_latest_nonempty_top50_cache(max_age_hours: float = 24.0) -> Optional[dict[str, Any]]:
    if not TOP50_CACHE.exists():
        return None
    try:
        payload = json.loads(TOP50_CACHE.read_text(encoding="utf-8"))
        fetched_at = _parse_utc_datetime(payload.get("fetched_at"))
        if fetched_at is None:
            return None
        age = (_utc_now() - fetched_at).total_seconds()
        if age > max_age_hours * 3600:
            return None
        rows = payload.get("rows") or []
        if not rows:
            return None
        payload["cached"] = True
        payload["stale_fallback"] = True
        return payload
    except Exception as exc:
        logger.debug("latest nonempty top50 cache load failed: %s", exc)
        return None


def _daily_history(symbol: str, count: int = 120) -> tuple[pd.DataFrame, str]:
    hist = _longbridge_history(symbol, count)
    if not hist.empty:
        return hist, "longbridge"
    hist = _twelvedata_history(symbol, interval="1day", outputsize=count)
    if not hist.empty:
        return hist, "twelvedata"
    hist = _alpha_vantage_history(symbol, function="TIME_SERIES_DAILY_ADJUSTED")
    if not hist.empty:
        return hist.tail(count), "alphavantage"
    hist = _yahoo_history(symbol, period="6mo")
    return hist.tail(count) if not hist.empty else hist, "yahoo"


def _score_unusual_options(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return full
    scored = full.copy()
    for col in ("bid", "ask", "last", "volume", "oi", "iv_pct", "moneyness", "dte", "strike"):
        if col not in scored.columns:
            scored[col] = np.nan
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    bid = scored["bid"].fillna(0)
    ask = scored["ask"].fillna(0)
    last = scored["last"].fillna(0)
    mid = pd.Series(
        np.where(
            (bid > 0) & (ask > 0),
            (bid + ask) / 2,
            np.where(last > 0, last, np.where(ask > 0, ask, bid)),
        ),
        index=scored.index,
    )
    scored["mid"] = mid.round(4)

    spread = pd.Series(np.where((bid > 0) & (ask > 0), ask - bid, np.nan), index=scored.index)
    scored["spread"] = spread.round(4)
    if "spread_pct" not in scored.columns:
        scored["spread_pct"] = np.nan
    scored["spread_pct"] = pd.to_numeric(scored["spread_pct"], errors="coerce")
    scored["spread_pct"] = scored["spread_pct"].where(
        scored["spread_pct"].notna(),
        np.where((spread.notna()) & (mid > 0), (spread / mid * 100).round(2), np.nan),
    )

    volume = scored["volume"].fillna(0).clip(lower=0)
    oi = scored["oi"].fillna(0).clip(lower=0)
    ratio = volume / oi.where(oi > 0, 1.0)
    premium = mid.fillna(0).clip(lower=0) * volume * 100

    scored["vol_oi_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0).round(2)
    scored["premium"] = premium.replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

    liquidity_score = np.log10(volume + 1) * 10 + np.log10(oi + 1) * 4
    ratio_score = np.minimum(35, np.log1p(scored["vol_oi_ratio"]) * 16)
    premium_score = np.minimum(30, np.log10(scored["premium"] + 1) * 4.5)
    iv = scored["iv_pct"].fillna(0)
    iv_score = np.where((iv >= 25) & (iv <= 90), 8, np.where((iv > 90) & (iv <= 140), 4, 0))
    spread_penalty = np.minimum(28, scored["spread_pct"].fillna(12).clip(lower=0) * 0.55)
    moneyness_penalty = np.minimum(18, scored["moneyness"].abs().fillna(20) * 0.45)
    dte = scored["dte"].fillna(0)
    dte_penalty = np.where(dte <= 1, 8, np.where(dte > 60, 5, 0))

    scored["unusual_score"] = (
        liquidity_score
        + ratio_score
        + premium_score
        + iv_score
        - spread_penalty
        - moneyness_penalty
        - dte_penalty
    ).round(2)
    scored["score"] = scored["unusual_score"]
    return scored


def _has_option_activity_data(full: pd.DataFrame) -> bool:
    if full.empty:
        return False
    total = 0.0
    for col in ("volume", "oi", "last"):
        if col in full.columns:
            total += float(pd.to_numeric(full[col], errors="coerce").fillna(0).clip(lower=0).sum())
    return total > 0


def _attach_longbridge_depths(symbol: str, rows: pd.DataFrame, max_rows: int = 12) -> pd.DataFrame:
    if rows.empty or "lb_symbol" not in rows.columns:
        return rows
    enriched = rows.copy()
    for idx, row in enriched.head(max_rows).iterrows():
        lb_symbol = row.get("lb_symbol")
        if not lb_symbol:
            continue
        bid, ask = _longbridge_option_depth(str(lb_symbol))
        if bid is not None:
            enriched.at[idx, "bid"] = bid
        if ask is not None:
            enriched.at[idx, "ask"] = ask
    return _score_unusual_options(enriched)


def _enrich_longbridge_activity_subset(rows: pd.DataFrame, spot: float, max_rows: int = 160) -> pd.DataFrame:
    if rows.empty or "lb_symbol" not in rows.columns:
        return rows
    work = rows.copy()
    work["spot_ref"] = float(spot or 0)
    work["atm_distance"] = np.where(
        float(spot or 0) > 0,
        (pd.to_numeric(work.get("strike"), errors="coerce") - float(spot)).abs() / float(spot),
        np.nan,
    )
    work["atm_distance"] = pd.to_numeric(work["atm_distance"], errors="coerce").fillna(999.0)
    work["dte_rank"] = pd.to_numeric(work.get("dte"), errors="coerce").fillna(999.0)
    ranked = work.sort_values(["dte_rank", "atm_distance"])
    sampled = ranked.groupby(["expiry", "optionType"], as_index=False, group_keys=False).head(8)
    sampled = sampled.head(max_rows).copy()
    if sampled.empty:
        return rows

    quotes = _longbridge_option_quotes([s for s in sampled["lb_symbol"].dropna().unique().tolist() if s])
    if not quotes:
        return rows

    quote_rows = []
    for lb_symbol, quote in quotes.items():
        item = _extract_option_mark_from_quote(quote)
        item["lb_symbol"] = lb_symbol
        quote_rows.append(item)
    if not quote_rows:
        return rows

    quote_df = pd.DataFrame(quote_rows)
    enriched = sampled.drop(columns=[c for c in ("bid", "ask", "last", "volume", "oi", "iv_pct") if c in sampled.columns]).merge(
        quote_df,
        on="lb_symbol",
        how="left",
    )
    if "moneyness" not in enriched.columns and float(spot or 0) > 0:
        enriched["moneyness"] = ((pd.to_numeric(enriched["strike"], errors="coerce") - float(spot)) / float(spot) * 100).round(2)
    return enriched


def _unusual_direction_bonus(opt_type: str, tech_bias: str) -> float:
    opt_type = str(opt_type or "").upper()
    tech_bias = str(tech_bias or "").lower()
    if opt_type == "CALL" and "bullish" in tech_bias:
        return 5.0
    if opt_type == "PUT" and "bearish" in tech_bias:
        return 5.0
    if opt_type == "CALL" and tech_bias == "oversold":
        return 2.0
    if opt_type == "PUT" and tech_bias == "overbought":
        return 2.0
    if opt_type == "CALL" and "bearish" in tech_bias:
        return -3.0
    if opt_type == "PUT" and "bullish" in tech_bias:
        return -3.0
    return 0.0


def _unusual_reason_cn(row: dict[str, Any], plan: dict[str, Any], profile: dict[str, Any], tech_bias: str) -> str:
    symbol = row.get("symbol") or plan.get("symbol") or ""
    name = profile.get("long_name") or row.get("name") or symbol
    ratio = _num(row.get("vol_oi_ratio"))
    premium = _num(row.get("premium"))
    volume = int(_num(row.get("volume")))
    oi = int(_num(row.get("oi")))
    iv = row.get("iv_pct")
    spread = row.get("spread_pct")
    parts = [
        f"{name}：异常成交量 {volume:,} / OI {oi:,}，Vol/OI {ratio:.2f}x",
        f"名义权利金约 ${premium:,.0f}",
    ]
    if iv is not None:
        parts.append(f"IV {float(iv):.1f}%")
    if spread is not None:
        parts.append(f"点差 {float(spread):.1f}%")
    parts.append(f"技术面 {_tech_bias_cn(tech_bias)}")
    if ratio >= 3:
        parts.append("成交量显著压过持仓，可能是新开仓或短线资金集中进场")
    elif ratio >= 1:
        parts.append("成交量高于持仓基数，值得跟踪后续延续性")
    if premium >= 500000:
        parts.append("权利金规模较大，资金参与度更高")
    if plan.get("risk_reward") is not None:
        parts.append(f"计划RR {float(plan.get('risk_reward')):.2f}")
    return "；".join(parts)


def _unusual_analysis_cn(plan: dict[str, Any]) -> str:
    return (
        f"入场 {plan.get('entry') if plan.get('entry') is not None else '—'}，"
        f"止损 {plan.get('stop_loss') if plan.get('stop_loss') is not None else '—'}，"
        f"止盈 {plan.get('take_profit') if plan.get('take_profit') is not None else '—'}，"
        f"正股触发 {plan.get('underlying_trigger') if plan.get('underlying_trigger') is not None else '—'}，"
        f"失效 {plan.get('underlying_invalidation') if plan.get('underlying_invalidation') is not None else '—'}"
    )


def _prefilter_unusual_universe(
    universe: list[dict[str, Any]],
    bias: dict[str, float],
    max_symbols: int,
    prefilter_multiplier: float = UNUSUAL_PREFILTER_MULTIPLIER,
) -> list[dict[str, Any]]:
    symbols = [row.get("symbol") for row in universe if row.get("symbol")]
    lb_symbols = [f"{_symbol_root(sym)}.US" for sym in symbols]
    static_infos = _longbridge_static_info(lb_symbols)
    optionable = {getattr(info, "symbol", "") for info in static_infos if _is_optionable_static(info)}
    quote_targets = [symbol for symbol in lb_symbols if not optionable or symbol in optionable]
    quote_scan_cap = min(len(quote_targets), max(max_symbols * 3, int(max_symbols * max(1.5, prefilter_multiplier * 1.4))))
    quotes = _longbridge_quotes(quote_targets[:quote_scan_cap])

    prefiltered = []
    for row in universe:
        symbol = row.get("symbol")
        if not symbol:
            continue
        lb_symbol = f"{_symbol_root(symbol)}.US"
        if optionable and lb_symbol not in optionable:
            continue
        quote = quotes.get(lb_symbol)
        prefiltered.append(
            {
                **row,
                "lb_symbol": lb_symbol,
                "quote_volume": int(getattr(quote, "volume", 0) or 0) if quote is not None else 0,
                "quote_turnover": float(getattr(quote, "turnover", 0) or 0) if quote is not None else 0,
                "pre_score": _marketcap_ticker_score(row, quote, bias),
            }
        )

    if not prefiltered:
        prefiltered = [
            {
                **row,
                "lb_symbol": f"{_symbol_root(row.get('symbol', ''))}.US",
                "quote_volume": 0,
                "quote_turnover": 0,
                "pre_score": _marketcap_ticker_score(row, None, bias),
            }
            for row in universe
            if row.get("symbol")
        ]
    for item in prefiltered:
        quote_volume = max(0.0, float(item.get("quote_volume", 0) or 0))
        quote_turnover = max(0.0, float(item.get("quote_turnover", 0) or 0))
        item["prefilter_score"] = round(
            float(item.get("pre_score", 0) or 0)
            + min(10.0, math.log10(quote_volume + 1) * 1.8)
            + min(12.0, math.log10(quote_turnover + 1) * 0.75),
            2,
        )
    target = max(max_symbols, min(len(prefiltered), int(math.ceil(max_symbols * max(1.0, prefilter_multiplier)))))
    return sorted(prefiltered, key=lambda item: item.get("prefilter_score", item.get("pre_score", 0)), reverse=True)[:target]


def _scan_unusual_symbol(
    row: dict[str, Any],
    conclusion: dict[str, Any],
    min_dte: int,
    max_dte: int,
    otm_range: float,
    min_volume: int,
    min_premium: float,
    top_per_symbol: int,
) -> list[dict[str, Any]]:
    symbol = _normalize_symbol(row.get("symbol") or "")
    if not symbol:
        return []

    hist, history_source = _daily_history(symbol, 140)
    if hist.empty or "Close" not in hist:
        return []
    quote = _longbridge_quote(symbol)
    spot = _num((quote or {}).get("price"), _num(hist["Close"].iloc[-1], 0))
    if spot <= 0:
        return []

    close = hist["Close"].dropna()
    sma20 = _safe(close.tail(20).mean()) if not close.empty else None
    sma50 = _safe(close.tail(50).mean()) if len(close) >= 5 else None
    try:
        rsi = _calc_rsi(close) if len(close) >= 15 else 50.0
    except Exception:
        rsi = 50.0
    tech_bias = _tech_signal(spot, sma20, sma50, rsi)
    hv30 = _get_hv(hist, 30)

    full = _build_option_candidates(symbol, spot, min_dte, max_dte, "both", otm_range, enrich_quotes=True, allow_yfinance=False)
    if not full.empty and not _has_option_activity_data(full):
        enriched_subset = _enrich_longbridge_activity_subset(full, spot, max_rows=max(top_per_symbol * 40, 120))
        if not enriched_subset.empty and _has_option_activity_data(enriched_subset):
            full = enriched_subset
    if full.empty or not _has_option_activity_data(full):
        yahoo_full = _build_option_candidates(
            symbol,
            spot,
            min_dte,
            max_dte,
            "both",
            otm_range,
            enrich_quotes=True,
            allow_yfinance=True,
            allow_longbridge=False,
        )
        if not yahoo_full.empty:
            full = yahoo_full
    if full.empty:
        return []
    full = _score_unusual_options(full)
    full = _apply_option_factor_model(full, spot, hv30, tech_bias)
    full = full[
        (full["volume"].fillna(0) >= min_volume)
        | (full["premium"].fillna(0) >= min_premium)
        | (full["vol_oi_ratio"].fillna(0) >= 2.0)
    ].copy()
    if full.empty:
        fallback = _score_unusual_options(_apply_option_factor_model(full.copy() if not full.empty else _build_option_candidates(symbol, spot, min_dte, max_dte, "both", otm_range, enrich_quotes=False, allow_yfinance=False), spot, hv30, tech_bias))
        if not fallback.empty:
            fallback = fallback[
                (fallback["volume"].fillna(0) >= max(10, min_volume // 10))
                | (fallback["premium"].fillna(0) >= max(25000, min_premium * 0.25))
                | (fallback["oi"].fillna(0) >= 150)
                | (fallback["factor_score"].fillna(0) >= 8)
            ].copy()
            full = fallback
    if full.empty:
        return []

    top = full.nlargest(max(top_per_symbol * 4, 10), "unusual_score").copy()
    top = _attach_longbridge_depths(symbol, top, max_rows=max(top_per_symbol * 4, 10))
    top = _score_unusual_options(top)
    top = _apply_option_factor_model(top, spot, hv30, tech_bias).nlargest(top_per_symbol, "unusual_score").copy()
    if top.empty:
        return []

    greeks = top.apply(
        lambda opt_row: pd.Series(
            _bs_greeks(
                spot,
                float(opt_row["strike"]),
                float(opt_row["dte"]) / 365.0,
                RISK_FREE_RATE,
                (float(opt_row["iv_pct"]) / 100.0) if pd.notna(opt_row.get("iv_pct")) else 0,
                opt_row["optionType"],
            )
        ),
        axis=1,
    )
    top = pd.concat([top, greeks], axis=1)
    contracts = _option_contracts(symbol, top)
    iv_vals = full["iv_pct"].dropna()
    iv_rank = _safe((iv_vals.mean() - iv_vals.min()) / (iv_vals.max() - iv_vals.min()) * 100) if not iv_vals.empty and iv_vals.max() != iv_vals.min() else 50.0
    plans = _tag_trade_plans_source(_build_trade_plans(symbol, hist, contracts, spot, tech_bias, iv_rank), "unusual_daily")
    plan_by_contract = {plan.get("contract"): plan for plan in plans}
    profile = _cached_company_profile(symbol)
    news_items = _cached_recent_news(symbol, 2)

    results = []
    for contract, (_, opt_row) in zip(contracts, top.iterrows()):
        plan = plan_by_contract.get(contract.get("name"))
        if not plan:
            continue
        contract.update(
            {
                "mid": _safe(opt_row.get("mid"), 4),
                "vol_oi_ratio": _safe(opt_row.get("vol_oi_ratio")),
                "premium": _safe(opt_row.get("premium")),
                "unusual_score": _safe(opt_row.get("unusual_score")),
            }
        )
        plan.update(
            {
                "premium": contract.get("premium"),
                "vol_oi_ratio": contract.get("vol_oi_ratio"),
                "unusual_score": contract.get("unusual_score"),
                "spread_pct": contract.get("spread_pct"),
                "reason": "abnormal volume/OI + premium filter",
            }
        )
        style_bonus = _library_style_bonus(plan, conclusion)
        sim_bonus = _sim_learning_bonus(plan)
        learning_bonus = _adaptive_learning_bonus(plan)
        direction_bonus = _unusual_direction_bonus(plan.get("type"), tech_bias)
        final_score = _num(contract.get("unusual_score")) + style_bonus + sim_bonus + learning_bonus + direction_bonus + _num(row.get("pre_score")) * 0.04
        output = {
            "symbol": symbol,
            "name": row.get("name") or profile.get("long_name") or symbol,
            "sector": profile.get("sector"),
            "industry": profile.get("industry"),
            "market_cap": row.get("market_cap"),
            "price": row.get("price"),
            "spot": _safe(spot),
            "source": {"history": history_source, "spot": "longbridge" if quote else history_source},
            "tech_bias": tech_bias,
            "rsi": _safe(rsi),
            "hv30": hv30,
            "iv_rank": iv_rank,
            "news_headline": news_items[0].get("title") if news_items else "",
            "contract": contract,
            "best_plan": plan,
            "source_bucket": _plan_source_bucket(plan),
            "option_type": plan.get("type"),
            "expiry": plan.get("expiry"),
            "strike": plan.get("strike"),
            "dte": plan.get("dte"),
            "entry": plan.get("entry"),
            "stop_loss": plan.get("stop_loss"),
            "take_profit": plan.get("take_profit"),
            "underlying_trigger": plan.get("underlying_trigger"),
            "underlying_invalidation": plan.get("underlying_invalidation"),
            "rr": plan.get("risk_reward"),
            "volume": contract.get("volume"),
            "oi": contract.get("oi"),
            "iv_pct": contract.get("iv_pct"),
            "spread_pct": contract.get("spread_pct"),
            "vol_oi_ratio": contract.get("vol_oi_ratio"),
            "premium": contract.get("premium"),
            "unusual_score": contract.get("unusual_score"),
            "style_bonus": style_bonus,
            "sim_bonus": sim_bonus,
            "learning_bonus": learning_bonus,
            "direction_bonus": direction_bonus,
            "final_score": round(final_score, 2),
        }
        output["reason_cn"] = _unusual_reason_cn(output, plan, profile, tech_bias)
        output["analysis_cn"] = _unusual_analysis_cn(plan)
        results.append(output)
    return results


def _scan_daily_unusual_options(
    limit: int = 50,
    universe_size: int = 1000,
    refresh: bool = False,
    min_dte: int = 1,
    max_dte: int = 45,
    min_volume: int = 200,
    min_premium: float = 100000,
    max_symbols: Optional[int] = None,
    top_per_symbol: int = 3,
    use_library_bias: bool = True,
    cache_max_age_hours: float = 4.0,
    prefilter_multiplier: float = UNUSUAL_PREFILTER_MULTIPLIER,
    max_runtime_seconds: float = 35.0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 100))
    universe_size = max(20, min(int(universe_size), 1000))
    max_symbols = max(5, min(int(max_symbols or UNUSUAL_SCAN_SYMBOL_LIMIT), universe_size))
    min_dte = max(0, int(min_dte))
    max_dte = max(min_dte + 1, int(max_dte))
    top_per_symbol = max(1, min(int(top_per_symbol), UNUSUAL_MAX_TOP_PER_SYMBOL))
    prefilter_multiplier = max(1.0, min(float(prefilter_multiplier or UNUSUAL_PREFILTER_MULTIPLIER), 5.0))
    cache_key = {
        "limit": limit,
        "universe_size": universe_size,
        "min_dte": min_dte,
        "max_dte": max_dte,
        "min_volume": int(min_volume),
        "min_premium": float(min_premium),
        "max_symbols": max_symbols,
        "top_per_symbol": top_per_symbol,
        "use_library_bias": bool(use_library_bias),
        "prefilter_multiplier": prefilter_multiplier,
    }
    cached = _load_unusual_cache(cache_key, max_age_hours=cache_max_age_hours)
    latest_nonempty = _load_latest_nonempty_unusual_cache(max_age_hours=max(cache_max_age_hours, 24.0))
    if cached and not refresh:
        return cached
    if cached and refresh:
        fetched_at = _parse_utc_datetime(cached.get("fetched_at"))
        stale_seconds = (_utc_now() - fetched_at).total_seconds() if fetched_at else None
        if stale_seconds is not None and stale_seconds < 15 * 60:
            return cached

    profile = _chart_library_profile() if use_library_bias else {"tag_counts": {}}
    bias = _chart_library_bias(profile)
    conclusion = profile.get("conclusion", {}) if isinstance(profile, dict) else {}
    universe = _load_marketcap_universe(universe_size, refresh=refresh)
    if not universe:
        return {"error": "market cap universe unavailable"}

    prefiltered = _prefilter_unusual_universe(universe, bias, max_symbols, prefilter_multiplier=prefilter_multiplier)
    if not prefiltered:
        return {"error": "no symbols available for unusual option scan"}

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    scanned_symbols = 0
    started_at = _utc_now()
    otm_range = 0.18 if bias.get("dte_bias", 0) < 0 else 0.14
    target_rows = max(limit * 2, UNUSUAL_MIN_TARGET_ROWS)
    for item in prefiltered:
        symbol = item.get("symbol")
        try:
            scanned_symbols += 1
            rows.extend(
                _scan_unusual_symbol(
                    item,
                    conclusion,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    otm_range=otm_range,
                    min_volume=int(min_volume),
                    min_premium=float(min_premium),
                    top_per_symbol=top_per_symbol,
                )
            )
        except Exception as exc:
            logger.warning("%s unusual options scan failed: %s", symbol, exc)
            if symbol:
                errors.append({"symbol": symbol, "error": str(exc)})
        elapsed = (_utc_now() - started_at).total_seconds()
        if elapsed >= max_runtime_seconds and rows:
            break
        if scanned_symbols >= max_symbols and len(rows) >= target_rows:
            break

    if not rows:
        relaxed_rows: list[dict[str, Any]] = []
        relaxed_min_volume = max(10, int(min_volume) // 4)
        relaxed_min_premium = max(25000.0, float(min_premium) * 0.25)
        relaxed_symbols = min(len(prefiltered), max(8, min(max_symbols, 12)))
        for item in prefiltered[:relaxed_symbols]:
            symbol = item.get("symbol")
            try:
                relaxed_rows.extend(
                    _scan_unusual_symbol(
                        item,
                        conclusion,
                        min_dte=min_dte,
                        max_dte=max_dte,
                        otm_range=otm_range,
                        min_volume=relaxed_min_volume,
                        min_premium=relaxed_min_premium,
                        top_per_symbol=max(top_per_symbol, 4),
                    )
                )
            except Exception as exc:
                logger.warning("%s unusual relaxed scan failed: %s", symbol, exc)
            elapsed = (_utc_now() - started_at).total_seconds()
            if elapsed >= max_runtime_seconds and relaxed_rows:
                break
        if relaxed_rows:
            rows = relaxed_rows

    rows = sorted(rows, key=lambda item: item.get("final_score", 0), reverse=True)[:limit]
    for idx, item in enumerate(rows, start=1):
        item["rank"] = idx
    for item in rows[: min(len(rows), 20)]:
        plan = item.get("best_plan") if isinstance(item.get("best_plan"), dict) else None
        if plan:
            _record_signal_candidates(str(item.get("symbol") or ""), _num(item.get("spot")), [plan], str(item.get("tech_bias") or ""), source="unusual_daily")

    payload = {
        "fetched_at": _utc_now().isoformat(),
        "timestamp": _now_et_iso(),
        "cache_key": cache_key,
        "cached": False,
        "universe_size": len(universe),
        "prefiltered_size": len(prefiltered),
        "scanned_symbols": scanned_symbols,
        "returned": len(rows),
        "profile": profile,
        "conclusion": conclusion,
        "bias": bias,
        "errors": errors[:20],
        "rows": rows,
        "partial": bool((_utc_now() - started_at).total_seconds() >= max_runtime_seconds),
    }
    try:
        market_lines = [
            f"- returned: {len(rows)} / scanned_symbols: {scanned_symbols} / prefiltered_size: {len(prefiltered)}",
            f"- conclusion: direction {conclusion.get('direction', 'neutral')} / structure {conclusion.get('structure', 'both')} / dte {conclusion.get('dte_style', 'mixed')}",
        ]
        for item in rows[:6]:
            market_lines.append(
                f"- {item.get('symbol')}: {item.get('option_type')} score {item.get('final_score')} premium {_short_money(item.get('premium'))}"
            )
        _append_market_memory("unusual_daily", market_lines, timestamp=payload.get("timestamp"))
    except Exception as exc:
        logger.debug("market memory append skipped for unusual scan: %s", exc)
    if not rows and latest_nonempty:
        latest_nonempty["fallback_reason"] = "current_scan_empty"
        latest_nonempty["timestamp"] = _now_et_iso()
        latest_nonempty["cache_key"] = cache_key
        return latest_nonempty
    _save_unusual_cache(payload)
    return payload


@app.route("/api/scan", methods=["POST"])
def scan():
    body = request.get_json(silent=True) or {}
    symbol = _normalize_symbol(body.get("symbol", "SPY"))
    min_dte = int(body.get("min_dte", MIN_DTE))
    max_dte = int(body.get("max_dte", MAX_DTE))
    opt_filter = str(body.get("opt_type", "call")).lower()
    otm_range = float(body.get("otm_range", OTM_RANGE_PCT))
    top_n = int(body.get("top_n", 8))
    tw_key = (body.get("twelvedata_api_key") or TWELVEDATA_API_KEY).strip()
    av_key = (body.get("alpha_vantage_api_key") or ALPHAVANTAGE_API_KEY).strip()
    ai_prompt = body.get("ai_prompt") or ""
    enable_ai = bool(body.get("enable_ai", True))
    ai_mode = str(body.get("ai_mode", "single")).strip().lower()
    ai_providers = _collect_ai_providers(body)
    enable_webull_market_data = bool(body.get("enable_webull_market_data", False))
    webull_cfg = _merge_webull_config(_load_webull_config(), body.get("webull_settings") if isinstance(body.get("webull_settings"), dict) else None)
    webull_market_status = {"enabled": enable_webull_market_data, "used_for": [], "probe": {}}

    try:
        library_profile = _chart_library_profile()
        library_conclusion = library_profile.get("conclusion", {})
        hist90 = _empty_df()
        history_source = ""
        if enable_webull_market_data:
            webull_market_status["probe"] = _webull_market_probe(webull_cfg, symbol)
            hist90 = _webull_history(symbol, 120, WebullTimespan.D.name if WebullTimespan is not None else "D", webull_cfg)
            if not hist90.empty:
                history_source = "webull"
                webull_market_status["used_for"].append("history")
        if hist90.empty:
            hist90 = _longbridge_history(symbol, 120)
            history_source = "longbridge"
        if hist90.empty:
            hist90 = _twelvedata_history(symbol, interval="1day", outputsize=120, api_key=tw_key)
            history_source = "twelvedata"
        if hist90.empty:
            hist90 = _alpha_vantage_history(symbol, function="TIME_SERIES_DAILY_ADJUSTED", api_key=av_key)
            history_source = "alphavantage"
        if hist90.empty:
            hist90 = _yahoo_history(symbol, period="90d")
            history_source = "yahoo"
        if hist90.empty:
            return jsonify({"error": f"无法获取 {symbol} 历史数据"}), 400

        yahoo_spot = _safe(hist90["Close"].iloc[-1])
        webull_quote = _webull_quote(symbol, webull_cfg) if enable_webull_market_data else None
        lb_quote = _longbridge_quote(symbol)
        if webull_quote and webull_quote.get("price") is not None:
            spot = webull_quote["price"]
            webull_market_status["used_for"].append("spot")
        else:
            spot = lb_quote["price"] if lb_quote and lb_quote.get("price") else yahoo_spot
        if spot is None:
            return jsonify({"error": f"无法获取 {symbol} 当前价格"}), 400

        hv30 = _get_hv(hist90, 30)
        hv10 = _get_hv(hist90, 10)

        hist1y = _webull_history(symbol, 400, WebullTimespan.D.name if WebullTimespan is not None else "D", webull_cfg) if enable_webull_market_data else _empty_df()
        if not hist1y.empty and "history" not in webull_market_status["used_for"]:
            webull_market_status["used_for"].append("history")
        if hist1y.empty:
            hist1y = _longbridge_history(symbol, 400)
        if hist1y.empty:
            hist1y = _twelvedata_history(symbol, interval="1day", outputsize=400, api_key=tw_key)
        if hist1y.empty:
            hist1y = _alpha_vantage_history(symbol, function="TIME_SERIES_DAILY_ADJUSTED", api_key=av_key)
        if hist1y.empty:
            hist1y = _yahoo_history(symbol, period="1y")
        wk52h = _safe(hist1y["High"].max()) if not hist1y.empty else None
        wk52l = _safe(hist1y["Low"].min()) if not hist1y.empty else None

        sma20 = _safe(hist90["Close"].tail(20).mean())
        sma50 = _safe(hist90["Close"].tail(50).mean())
        rsi = _calc_rsi(hist90["Close"])
        tech_bias = _tech_signal(spot, sma20, sma50, rsi)

        effective_filter = "both" if _should_expand_scan_filter(opt_filter, tech_bias) else opt_filter
        full = _build_option_candidates(symbol, spot, min_dte, max_dte, effective_filter, otm_range)
        if not full.empty and not _has_option_activity_data(full):
            enriched_subset = _enrich_longbridge_activity_subset(full, spot, max_rows=max(top_n * 40, 120))
            if not enriched_subset.empty and _has_option_activity_data(enriched_subset):
                full = enriched_subset
        if full.empty:
            return jsonify({"error": "没有符合条件的期权合约"}), 400
        options_source = "longbridge" if "lb_symbol" in full.columns else "yahoo"

        full = _score_options(full)
        full = _score_unusual_options(full)
        full = _apply_option_factor_model(full, spot, hv30, tech_bias)
        top = _select_scan_rows(full, top_n, opt_filter, tech_bias)
        if "lb_symbol" in top.columns:
            bids = []
            asks = []
            for _, row in top.iterrows():
                bid, ask = _longbridge_option_depth(row.get("lb_symbol"))
                bids.append(bid)
                asks.append(ask)
            top["bid"] = pd.to_numeric(pd.Series(bids, index=top.index), errors="coerce")
            top["ask"] = pd.to_numeric(pd.Series(asks, index=top.index), errors="coerce")
            top["spread"] = top["ask"] - top["bid"]
            top["spread_pct"] = np.where(
                top["last"].fillna(0) > 0,
                (top["spread"] / top["last"] * 100).round(2),
                np.nan,
            )
        greeks = top.apply(
            lambda row: pd.Series(
                _bs_greeks(
                    spot,
                    float(row["strike"]),
                    float(row["dte"]) / 365.0,
                    RISK_FREE_RATE,
                    (float(row["iv_pct"]) / 100.0) if pd.notna(row.get("iv_pct")) else 0,
                    row["optionType"],
                )
            ),
            axis=1,
        )
        top = pd.concat([top, greeks], axis=1)
        top = top.sort_values("scan_score" if "scan_score" in top.columns else "score", ascending=False)

        contracts = _option_contracts(symbol, top)
        iv_vals = full["iv_pct"].dropna()
        atm_idx = (full["strike"] - spot).abs().idxmin()
        atm_iv = _safe(full.loc[atm_idx, "iv_pct"])
        if not iv_vals.empty and iv_vals.max() != iv_vals.min():
            iv_rank = _safe((iv_vals.mean() - iv_vals.min()) / (iv_vals.max() - iv_vals.min()) * 100)
        else:
            iv_rank = 50.0
        call_iv = full[full["optionType"] == "call"]["iv_pct"].mean()
        put_iv = full[full["optionType"] == "put"]["iv_pct"].mean()
        iv_skew = _safe(put_iv - call_iv)

        signal = _iv_signal(iv_rank, iv_skew)
        trade_plans = _tag_trade_plans_source(_build_trade_plans(symbol, hist90, contracts, spot, tech_bias, iv_rank), "scan")
        _record_signal_candidates(symbol, spot, trade_plans, tech_bias, source="scan")
        strategy = _strategy_text(symbol, contracts, spot, tech_bias, atm_iv, iv_rank, trade_plans=trade_plans, library_conclusion=library_conclusion)
        strategy["ai_mode"] = ai_mode
        strategy["ai_provider"] = body.get("ai_provider", "deepseek")
        strategy["sim_learning"] = _sim_learning_profile()
        strategy["adaptive_learning"] = _adaptive_learning_profile()
        strategy["knowledge_context"] = _knowledge_context_for_prompt(max_chars=1200)
        strategy["factor_summary"] = {
            "hv30": hv30,
            "front_atm_iv": atm_iv,
            "top_factor_scores": [plan.get("factor_score") for plan in trade_plans[:3]],
        }
        if enable_ai:
            ai_text_prompt = (
                f"标的: {symbol}\n"
                f"现价: {spot}\n"
                f"技术偏向: {tech_bias}\n"
                f"IV Rank: {iv_rank}\n"
                f"ATM IV: {atm_iv}\n"
                f"经验库结论: {json.dumps(library_conclusion, ensure_ascii=False)}\n"
                f"自主学习知识库: {strategy['knowledge_context']}\n"
                f"候选交易: {json.dumps(trade_plans[:3], ensure_ascii=False)}\n"
                f"用户补充: {ai_prompt}\n"
                "请输出中文结构化建议，格式分为：1. 总结 2. 主要理由 3. 主方案 4. 备选方案 5. 失效条件 6. 仓位与执行纪律。不要空话。"
            )
            discussion = _run_ai_discussion(ai_text_prompt, ai_providers, mode=ai_mode, primary_provider=body.get("ai_provider"))
            if discussion.get("final"):
                strategy["ai_brief"] = discussion["final"]
            if discussion.get("rounds"):
                strategy["ai_rounds"] = discussion["rounds"]
            strategy["ai_mode"] = discussion.get("mode", ai_mode)
            strategy["ai_provider"] = discussion.get("chosen_provider")

        return jsonify(
            {
                "symbol": symbol,
                "spot": spot,
                "source": {
                    "spot": "webull" if webull_quote and webull_quote.get("price") is not None else ("longbridge" if lb_quote else "yahoo"),
                    "history": history_source,
                    "options": options_source,
                },
                "scan_filter": {
                    "requested": opt_filter,
                    "effective": effective_filter,
                    "tech_direction": _tech_direction_from_bias(tech_bias),
                    "included_types": sorted({str(item.get("type") or "").upper() for item in contracts if item.get("type")}),
                },
                "webull_market": webull_market_status,
                "hv30": hv30,
                "hv10": hv10,
                "wk52h": wk52h,
                "wk52l": wk52l,
                "sma20": sma20,
                "sma50": sma50,
                "rsi": _safe(rsi),
                "tech_bias": tech_bias,
                "atm_iv": atm_iv,
                "iv_rank": iv_rank,
                "iv_skew": iv_skew,
                "signal": signal,
                "strategy": strategy,
                "trade_plans": trade_plans,
                "contracts": contracts,
                "library_profile": library_profile,
                "library_conclusion": library_conclusion,
                "learning_profile": _adaptive_learning_profile(),
                "ai_mode": ai_mode,
                "ai_providers": [p.get("name") for p in ai_providers],
            }
        )
    except Exception as exc:
        logger.exception("scan error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chart", methods=["GET"])
def chart():
    symbol = _normalize_symbol(request.args.get("symbol", "SPY"))
    kd = int(request.args.get("kdays", 60))
    prefer_webull = str(request.args.get("prefer_webull_market", "")).strip().lower() in {"1", "true", "yes", "on"}
    webull_cfg = _load_webull_config()

    try:
        hist = _webull_history(symbol, max(kd * 2, 120), WebullTimespan.D.name if WebullTimespan is not None else "D", webull_cfg) if prefer_webull else _empty_df()
        if hist.empty:
            hist = _longbridge_history(symbol, max(kd * 2, 120))
        if not hist.empty:
            hist = hist.tail(kd)
        if hist.empty:
            hist = _twelvedata_history(symbol, interval="1day", outputsize=max(kd * 2, 120))
        if not hist.empty:
            hist = hist.tail(kd)
        if hist.empty:
            hist = _alpha_vantage_history(symbol, function="TIME_SERIES_DAILY_ADJUSTED")
        if not hist.empty:
            hist = hist.tail(kd)
        if hist.empty:
            hist = _yahoo_history(symbol, period=f"{kd}d")
        if hist.empty:
            return jsonify({"error": f"无法获取 {symbol} 图表数据"}), 400

        daily = [
            {
                "t": ts.strftime("%Y-%m-%d"),
                "o": _safe(row["Open"]),
                "h": _safe(row["High"]),
                "l": _safe(row["Low"]),
                "c": _safe(row["Close"]),
                "v": int(row["Volume"]),
            }
            for ts, row in hist.iterrows()
        ]

        intra = _webull_history(symbol, 200, WebullTimespan.M5.name if WebullTimespan is not None else "M5", webull_cfg) if prefer_webull else _empty_df()
        if intra.empty:
            intra = _longbridge_intraday(symbol, 200)
        if intra.empty:
            intra = _twelvedata_history(symbol, interval="5min", outputsize=200)
        if intra.empty:
            intra = _alpha_vantage_history(symbol, function="TIME_SERIES_INTRADAY", interval="5min")
        if intra.empty:
            intra = _yahoo_history(symbol, period="1d", interval="5m")
        vwap_df = _compute_vwap(intra) if not intra.empty else _empty_df()
        intraday = [
            {
                "t": ts.strftime("%H:%M"),
                "c": _safe(row["Close"]),
                "v": int(row["Volume"]),
                "vwap": _safe(row["vwap"]),
            }
            for ts, row in vwap_df.iterrows()
        ] if not vwap_df.empty else []

        return jsonify({"daily": daily, "intraday": intraday})
    except Exception as exc:
        logger.exception("chart error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/vwap_pct", methods=["GET"])
def vwap_pct():
    symbol = _normalize_symbol(request.args.get("symbol", "SPY"))
    prefer_webull = str(request.args.get("prefer_webull_market", "")).strip().lower() in {"1", "true", "yes", "on"}
    webull_cfg = _load_webull_config()
    try:
        intra = _webull_history(symbol, 200, WebullTimespan.M5.name if WebullTimespan is not None else "M5", webull_cfg) if prefer_webull else _empty_df()
        if intra.empty:
            intra = _longbridge_intraday(symbol, 200)
        if intra.empty:
            intra = _twelvedata_history(symbol, interval="5min", outputsize=200)
        if intra.empty:
            intra = _alpha_vantage_history(symbol, function="TIME_SERIES_INTRADAY", interval="5min")
        if intra.empty:
            intra = _yahoo_history(symbol, period="1d", interval="5m")
        if intra.empty:
            return jsonify({"vwap_pct": None})
        df = _compute_vwap(intra)
        spot = float(df["Close"].iloc[-1])
        vwap = float(df["vwap"].iloc[-1])
        pct = _safe((spot - vwap) / vwap * 100)
        return jsonify({"vwap": _safe(vwap), "vwap_pct": pct, "spot": _safe(spot)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/webull/market_status", methods=["GET"])
def webull_market_status():
    symbol = _normalize_symbol(request.args.get("symbol", "AAPL"))
    cfg = _load_webull_config()
    return jsonify(_webull_market_probe(cfg, symbol))


@app.route("/api/market_temperature", methods=["GET"])
def market_temperature():
    market = _normalize_symbol(request.args.get("market", "US"))
    result = _longbridge_market_temperature(market)
    if result is None:
        return jsonify({"error": "Longbridge market temperature unavailable"}), 503
    return jsonify(result)


@app.route("/api/library/files", methods=["GET"])
def library_files():
    return jsonify(
        {
            "files": _chart_library_files(),
            "profile": _chart_library_profile(include_ai_summary=True),
        }
    )


@app.route("/api/library/profile", methods=["GET"])
def library_profile():
    profile = _chart_library_profile(include_ai_summary=True)
    return jsonify(
        {
            "profile": profile,
            "bias": _chart_library_bias(profile),
        }
    )


@app.route("/api/library/upload", methods=["POST"])
def library_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files"}), 400

    saved = []
    for item in files:
        filename = secure_filename(item.filename or "")
        if not filename or not _is_image_file(Path(filename)):
            continue
        target = CHART_LIBRARY_INBOX / filename
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        idx = 1
        while target.exists():
            target = CHART_LIBRARY_INBOX / f"{stem}_{idx}{suffix}"
            idx += 1
        item.save(target)
        saved.append(target.name)
    return jsonify({"saved": saved})


@app.route("/api/library/update", methods=["POST"])
def library_update():
    body = request.get_json(silent=True) or {}
    filename = secure_filename(body.get("filename", ""))
    if not filename:
        return jsonify({"error": "filename required"}), 400
    payload = {
        "tags": body.get("tags", []),
        "notes": body.get("notes", ""),
        "symbol": body.get("symbol", ""),
        "status": body.get("status", "unlabeled"),
    }
    entry = _update_chart_manifest(filename, payload)
    return jsonify({"filename": filename, "meta": entry})


@app.route("/api/library/batch_update", methods=["POST"])
def library_batch_update():
    body = request.get_json(silent=True) or {}
    items = body.get("items", [])
    if not isinstance(items, list) or not items:
        return jsonify({"error": "items required"}), 400

    saved = []
    for item in items:
        if not isinstance(item, dict):
            continue
        filename = secure_filename(item.get("filename", ""))
        if not filename:
            continue
        payload = {
            "tags": item.get("tags", []),
            "notes": item.get("notes", ""),
            "symbol": item.get("symbol", ""),
            "status": item.get("status", "unlabeled"),
            "ocr_text": item.get("ocr_text", ""),
            "ocr_summary": item.get("ocr_summary", ""),
            "auto_tags": item.get("auto_tags", []),
            "source_folder": item.get("source_folder", ""),
            "import_mode": item.get("import_mode", "local_folder"),
        }
        entry = _update_chart_manifest(filename, payload)
        saved.append({"filename": filename, "meta": entry})

    return jsonify({"saved": saved, "count": len(saved)})


@app.route("/api/library/file/<path:filename>", methods=["GET"])
def library_file(filename):
    safe_name = secure_filename(filename)
    for folder in (CHART_LIBRARY_INBOX, CHART_LIBRARY_ARCHIVE):
        candidate = folder / safe_name
        if candidate.exists():
            return send_from_directory(folder, safe_name)
    return jsonify({"error": "file not found"}), 404


@app.route("/api/sim/portfolio", methods=["GET"])
def sim_portfolio():
    state = _refresh_sim_state(_load_sim_state())
    return jsonify(
        {
            "summary": _summarize_sim_state(state),
            "open_trades": [t for t in state.get("trades", []) if t.get("status") == "open"],
            "closed_trades": state.get("closed", [])[-20:],
            "updated_at": state.get("updated_at"),
            "learning": _adaptive_learning_profile(),
        }
    )


@app.route("/api/sim/open", methods=["POST"])
def sim_open():
    body = request.get_json(silent=True) or {}
    try:
        trade = _build_sim_trade(body)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    state = _load_sim_state()
    state.setdefault("trades", []).append(trade)
    state["updated_at"] = _now_et_iso()
    _save_sim_state(state)
    state = _refresh_sim_state(_load_sim_state())
    return jsonify({"trade": trade, "summary": _summarize_sim_state(state)})


@app.route("/api/sim/close", methods=["POST"])
def sim_close():
    body = request.get_json(silent=True) or {}
    trade_id = str(body.get("id") or "").strip()
    if not trade_id:
        return jsonify({"error": "id required"}), 400
    close_price = body.get("close_price")
    if close_price is not None:
        try:
            close_price = float(close_price)
        except Exception:
            return jsonify({"error": "close_price invalid"}), 400
    note = str(body.get("note") or "")
    state = _load_sim_state()
    try:
        trade = _close_sim_trade(state, trade_id, close_price=close_price, note=note)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    state["updated_at"] = _now_et_iso()
    _save_sim_state(state)
    state = _refresh_sim_state(_load_sim_state())
    return jsonify({"trade": trade, "summary": _summarize_sim_state(state)})


@app.route("/api/webull/settings", methods=["GET", "POST"])
def webull_settings():
    if request.method == "GET":
        cfg = _load_webull_config()
        return jsonify({"config": _mask_webull_config(cfg), "state": _load_webull_state()})

    body = request.get_json(silent=True) or {}
    existing = _load_webull_config()
    merged = _merge_webull_config(existing, body)
    try:
        saved = _save_webull_config(merged)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"config": _mask_webull_config(saved), "state": _load_webull_state()})


@app.route("/api/webull/status", methods=["GET"])
def webull_status():
    cfg = _load_webull_config()
    snapshot = _webull_snapshot(cfg)
    return jsonify(snapshot)


@app.route("/api/webull/accounts", methods=["GET"])
def webull_accounts():
    cfg = _load_webull_config()
    try:
        _, trade_client = _webull_client(cfg)
        resp = trade_client.account_v2.get_account_list()
        payload = _webull_response_payload(resp)
        accounts = []
        for item in _webull_walk_dicts(payload):
            aid = item.get("account_id") or item.get("accountId") or item.get("id")
            if not aid:
                continue
            accounts.append(
                {
                    "account_id": str(aid),
                    "account_number": str(item.get("account_number") or item.get("accountNo") or ""),
                    "name": item.get("account_name") or item.get("accountName") or item.get("name") or "",
                    "type": item.get("account_type") or item.get("accountType") or item.get("type") or "",
                    "raw": item,
                }
            )
        return jsonify({"accounts": accounts[:10], "raw": payload})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/webull/positions", methods=["GET"])
def webull_positions():
    cfg = _load_webull_config()
    try:
        _, trade_client = _webull_client(cfg)
        accounts_resp = trade_client.account_v2.get_account_list()
        accounts_payload = _webull_response_payload(accounts_resp)
        account_id = _webull_select_account_id(cfg, accounts_payload)
        if not account_id:
            raise RuntimeError("Webull account_id not found")
        pos_resp = trade_client.account_v2.get_account_position(account_id)
        bal_resp = trade_client.account_v2.get_account_balance(account_id)
        pos_payload = _webull_response_payload(pos_resp)
        bal_payload = _webull_response_payload(bal_resp)
        positions, open_option_positions, used_capital = _webull_count_option_positions(pos_payload)
        balance = _webull_balance_summary(bal_payload)
        return jsonify(
            {
                "account_id": account_id,
                "positions": positions,
                "open_option_positions": open_option_positions,
                "used_capital": used_capital,
                "balance": balance,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/webull/execute", methods=["POST"])
def webull_execute():
    body = request.get_json(silent=True) or {}
    cfg = _merge_webull_config(_load_webull_config(), body.get("settings") if isinstance(body.get("settings"), dict) else None)
    if not cfg.get("enabled") and not body.get("force", False):
        return jsonify({"error": "Webull auto trading is disabled"}), 400

    plan = body.get("trade_plan") or body.get("plan")
    trade_plans = body.get("trade_plans") if isinstance(body.get("trade_plans"), list) else []
    notes = str(body.get("strategy_notes") or cfg.get("strategy_notes") or "").strip()
    selected = None
    ranked = []
    if trade_plans:
        selected, ranked = _webull_rank_live_plan(trade_plans, notes)
    elif isinstance(plan, dict):
        selected = plan
    if not selected:
        return jsonify({"error": "No eligible option plan supplied"}), 400

    try:
        result = _webull_place_option_order(
            cfg,
            selected,
            preview_only=bool(body.get("preview_only", False)),
            prevent_duplicate=bool(body.get("prevent_duplicate", False)),
        )
        result["selected"] = selected
        result["ranked"] = ranked[:10]
        result["strategy_notes"] = notes
        result["config"] = _mask_webull_config(cfg)
        return jsonify(result)
    except Exception as exc:
        cfg["last_error"] = str(exc)
        try:
            _save_webull_config(cfg)
        except Exception:
            pass
        state = _load_webull_state()
        executions = state.setdefault("executions", [])
        executions.append(
            {
                "ts": _now_et_iso(),
                "error": str(exc),
                "selected": selected,
                "notes": notes,
            }
        )
        state["executions"] = executions[-200:]
        _save_webull_state(state)
        return jsonify({"error": str(exc), "selected": selected, "ranked": ranked[:10]}), 400


def _analyze_marketcap_universe(
    limit: int = 50,
    universe_size: int = 1000,
    refresh: bool = False,
    use_library_bias: bool = True,
    cache_max_age_hours: float = 4.0,
    max_runtime_seconds: float = 35.0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 100))
    universe_size = max(20, min(int(universe_size), 1000))
    cache_key = {
        "limit": limit,
        "universe_size": universe_size,
        "use_library_bias": bool(use_library_bias),
    }
    cached = _load_top50_cache(cache_key, max_age_hours=cache_max_age_hours)
    if cached and not refresh:
        return cached
    if cached and refresh:
        fetched_at = _parse_utc_datetime(cached.get("fetched_at"))
        stale_seconds = (_utc_now() - fetched_at).total_seconds() if fetched_at else None
        if stale_seconds is not None and stale_seconds < 15 * 60:
            return cached

    profile = _chart_library_profile() if use_library_bias else {"tag_counts": {}}
    bias = _chart_library_bias(profile)
    conclusion = profile.get("conclusion", {}) if isinstance(profile, dict) else {}
    universe = _load_marketcap_universe(universe_size, refresh=refresh)
    if not universe:
        return {"error": "market cap universe unavailable"}

    symbols = [row["symbol"] for row in universe]
    static_infos = _longbridge_static_info([f"{_symbol_root(sym)}.US" for sym in symbols])
    optionable = {info.symbol for info in static_infos if _is_optionable_static(info)}
    quotes = _longbridge_quotes([f"{_symbol_root(sym)}.US" for sym in symbols if f"{_symbol_root(sym)}.US" in optionable])

    prefiltered = []
    for row in universe:
        lb_symbol = f"{_symbol_root(row['symbol'])}.US"
        if lb_symbol not in optionable:
            continue
        quote = quotes.get(lb_symbol)
        score = _marketcap_ticker_score(row, quote, bias)
        prefiltered.append(
            {
                **row,
                "lb_symbol": lb_symbol,
                "quote_volume": int(getattr(quote, "volume", 0) or 0) if quote is not None else 0,
                "quote_turnover": float(getattr(quote, "turnover", 0) or 0) if quote is not None else 0,
                "pre_score": score,
            }
        )

    if not prefiltered:
        return {"error": "no optionable symbols found"}

    prefiltered = sorted(prefiltered, key=lambda x: x["pre_score"], reverse=True)[: max(limit * 2, 60)]
    analyzed = []
    started_at = _utc_now()
    target_analyzed = max(limit * 2, min(12, len(prefiltered)))
    for row in prefiltered:
        symbol = row["symbol"]
        hist = _longbridge_history(symbol, 120)
        if hist.empty:
            hist = _twelvedata_history(symbol, interval="1day", outputsize=120)
        if hist.empty:
            hist = _alpha_vantage_history(symbol, function="TIME_SERIES_DAILY_ADJUSTED")
        if hist.empty:
            continue

        spot = float(hist["Close"].iloc[-1])
        sma20 = _safe(hist["Close"].tail(20).mean())
        sma50 = _safe(hist["Close"].tail(50).mean())
        rsi = _calc_rsi(hist["Close"])
        tech_bias = _tech_signal(spot, sma20, sma50, rsi)
        hv30 = _get_hv(hist, 30)
        iv_bias = bias.get("iv_bias", 0.0)
        dte_bias = bias.get("dte_bias", 0.0)
        otm_range = 0.12 if dte_bias >= 0 else 0.08
        full = _build_option_candidates(symbol, spot, 7, 45, "both", otm_range, enrich_quotes=False, allow_yfinance=False)
        if full.empty:
            continue
        full = _score_options(full)
        full = _apply_option_factor_model(full, spot, hv30, tech_bias)
        top = full.nlargest(1, "score").copy()
        if "lb_symbol" in top.columns:
            bids = []
            asks = []
            for _, opt_row in top.iterrows():
                bid, ask = _longbridge_option_depth(opt_row.get("lb_symbol"))
                bids.append(bid)
                asks.append(ask)
            top["bid"] = pd.to_numeric(pd.Series(bids, index=top.index), errors="coerce")
            top["ask"] = pd.to_numeric(pd.Series(asks, index=top.index), errors="coerce")
            top["spread"] = top["ask"] - top["bid"]
            top["spread_pct"] = np.where(top["last"].fillna(0) > 0, (top["spread"] / top["last"] * 100).round(2), np.nan)

        greeks = top.apply(
            lambda opt_row: pd.Series(
                _bs_greeks(
                    spot,
                    float(opt_row["strike"]),
                    float(opt_row["dte"]) / 365.0,
                    RISK_FREE_RATE,
                    (float(opt_row["iv_pct"]) / 100.0) if pd.notna(opt_row.get("iv_pct")) else 0,
                    opt_row["optionType"],
                )
            ),
            axis=1,
        )
        top = pd.concat([top, greeks], axis=1)
        contracts = _option_contracts(symbol, top)
        plans = _tag_trade_plans_source(_build_trade_plans(symbol, hist, contracts, spot, tech_bias, _safe(top["iv_pct"].mean())), "pool")
        if not plans:
            continue
        best_plan = max(plans, key=lambda p: float(p.get("risk_reward") or 0))
        style_bonus = _library_style_bonus(best_plan, conclusion)
        sim_bonus = _sim_learning_bonus(best_plan)
        learning_bonus = _adaptive_learning_bonus(best_plan)
        final_score = float(row["pre_score"]) + float(best_plan.get("risk_reward") or 0) * 5 + float(best_plan.get("score") or 0) * 0.01 + style_bonus + sim_bonus + learning_bonus
        analyzed.append(
            {
                "rank": len(analyzed) + 1,
                "symbol": symbol,
                "name": row["name"],
                "market_cap": row["market_cap"],
                "price": row["price"],
                "change_pct": row["change_pct"],
                "spot": spot,
                "tech_bias": tech_bias,
                "hv30": hv30,
                "rsi": _safe(rsi),
                "iv_rank": _safe((full["iv_pct"].mean() - full["iv_pct"].min()) / (full["iv_pct"].max() - full["iv_pct"].min()) * 100) if full["iv_pct"].max() != full["iv_pct"].min() else 50.0,
                "best_plan": best_plan,
                "source_bucket": _plan_source_bucket(best_plan),
                "final_score": round(final_score, 2),
                "style_bonus": style_bonus,
                "sim_bonus": sim_bonus,
                "learning_bonus": learning_bonus,
                "profile_bias": bias,
            }
        )
        elapsed = (_utc_now() - started_at).total_seconds()
        if len(analyzed) >= target_analyzed:
            break
        if elapsed >= max_runtime_seconds and analyzed:
            break

    analyzed = sorted(analyzed, key=lambda x: x["final_score"], reverse=True)[:limit]
    for idx, item in enumerate(analyzed, start=1):
        item["rank"] = idx
        item["entry"] = item.get("best_plan", {}).get("entry")
        item["stop_loss"] = item.get("best_plan", {}).get("stop_loss")
        item["take_profit"] = item.get("best_plan", {}).get("take_profit")
        item["underlying_trigger"] = item.get("best_plan", {}).get("underlying_trigger")
        item["underlying_invalidation"] = item.get("best_plan", {}).get("underlying_invalidation")
        item["rr"] = item.get("best_plan", {}).get("risk_reward")
        profile = _cached_company_profile(item["symbol"])
        news_items = _cached_recent_news(item["symbol"], 3)
        item["sector"] = profile.get("sector")
        item["industry"] = profile.get("industry")
        item["company_name"] = profile.get("long_name") or item.get("name")
        item["news_headline"] = news_items[0].get("title") if news_items else ""
        item["reason_cn"] = _top50_reason_cn(
            item,
            item.get("best_plan", {}),
            conclusion,
            float(item.get("style_bonus") or 0),
            float(item.get("sim_bonus") or 0),
            profile=profile,
            news_items=news_items,
        )
        item["analysis_cn"] = (
            f"入场 {item['entry'] if item['entry'] is not None else '—'}，"
            f"止损 {item['stop_loss'] if item['stop_loss'] is not None else '—'}，"
            f"止盈 {item['take_profit'] if item['take_profit'] is not None else '—'}，"
            f"触发位 {item['underlying_trigger'] if item['underlying_trigger'] is not None else '—'}，"
            f"失效位 {item['underlying_invalidation'] if item['underlying_invalidation'] is not None else '—'}"
        )
    for item in analyzed[: min(len(analyzed), 12)]:
        plan = item.get("best_plan") if isinstance(item.get("best_plan"), dict) else None
        if plan:
            _record_signal_candidates(str(item.get("symbol") or ""), _num(item.get("spot")), [plan], str(item.get("tech_bias") or ""), source="top50_universe")
    payload = {
        "fetched_at": _utc_now().isoformat(),
        "timestamp": _now_et_iso(),
        "cache_key": cache_key,
        "universe_size": len(universe),
        "analyzed_size": len(prefiltered),
        "returned": len(analyzed),
        "profile": profile,
        "conclusion": conclusion,
        "bias": bias,
        "rows": analyzed,
        "partial": bool((_utc_now() - started_at).total_seconds() >= max_runtime_seconds),
    }
    try:
        market_lines = [
            f"- returned: {len(analyzed)} / analyzed_size: {len(prefiltered)} / universe_size: {len(universe)}",
            f"- conclusion: direction {conclusion.get('direction', 'neutral')} / structure {conclusion.get('structure', 'both')} / dte {conclusion.get('dte_style', 'mixed')}",
        ]
        for item in analyzed[:8]:
            plan = item.get("best_plan", {}) if isinstance(item.get("best_plan"), dict) else {}
            market_lines.append(
                f"- {item.get('symbol')}: {plan.get('type') or item.get('profile_bias') or '?'} score {item.get('final_score')} sector {item.get('sector') or 'Unknown'}"
            )
        _append_market_memory("top50_universe", market_lines, timestamp=payload.get("timestamp"))
    except Exception as exc:
        logger.debug("market memory append skipped for top50: %s", exc)
    _save_top50_cache(payload)
    return payload


@app.route("/api/universe/top50", methods=["POST"])
def universe_top50():
    body = request.get_json(silent=True) or {}
    limit = int(body.get("limit", 50))
    universe_size = int(body.get("universe_size", UNIVERSE_SCAN_LIMIT))
    refresh = bool(body.get("refresh", False))
    use_library_bias = bool(body.get("use_library_bias", True))
    prefer_cached = bool(body.get("prefer_cached", False))
    if prefer_cached:
        cached = _load_top50_cache(
            {
                "limit": limit,
                "universe_size": universe_size,
                "use_library_bias": use_library_bias,
            },
            max_age_hours=float(body.get("cache_max_age_hours", 4)),
        )
        if cached:
            return jsonify(cached), 200
        latest_nonempty = _load_latest_nonempty_top50_cache(max_age_hours=max(float(body.get("cache_max_age_hours", 4)), 24.0))
        if latest_nonempty:
            latest_nonempty["timestamp"] = _now_et_iso()
            latest_nonempty["cache_key"] = {
                "limit": limit,
                "universe_size": universe_size,
                "use_library_bias": use_library_bias,
            }
            latest_nonempty["fallback_reason"] = "interactive_cached_snapshot"
            return jsonify(latest_nonempty), 200
    result = _analyze_marketcap_universe(
        limit=limit,
        universe_size=universe_size,
        refresh=refresh,
        use_library_bias=use_library_bias,
        cache_max_age_hours=float(body.get("cache_max_age_hours", 4)),
        max_runtime_seconds=float(body.get("max_runtime_seconds", 35)),
    )
    code = 200 if "error" not in result else 400
    return jsonify(result), code


@app.route("/api/unusual/daily", methods=["POST"])
def unusual_daily():
    body = request.get_json(silent=True) or {}
    prefer_cached = bool(body.get("prefer_cached", False))
    if prefer_cached:
        cache_key = {
            "limit": int(body.get("limit", 50)),
            "universe_size": int(body.get("universe_size", UNIVERSE_SCAN_LIMIT)),
            "min_dte": int(body.get("min_dte", 1)),
            "max_dte": int(body.get("max_dte", 45)),
            "min_volume": int(body.get("min_volume", 200)),
            "min_premium": float(body.get("min_premium", 100000)),
            "max_symbols": int(body.get("max_symbols", UNUSUAL_SCAN_SYMBOL_LIMIT)),
            "top_per_symbol": int(body.get("top_per_symbol", 3)),
            "use_library_bias": bool(body.get("use_library_bias", True)),
            "prefilter_multiplier": float(body.get("prefilter_multiplier", UNUSUAL_PREFILTER_MULTIPLIER)),
        }
        cached = _load_unusual_cache(cache_key, max_age_hours=float(body.get("cache_max_age_hours", 4)))
        if cached:
            return jsonify(cached), 200
        latest_nonempty = _load_latest_nonempty_unusual_cache(max_age_hours=max(float(body.get("cache_max_age_hours", 4)), 24.0))
        if latest_nonempty:
            latest_nonempty["timestamp"] = _now_et_iso()
            latest_nonempty["cache_key"] = cache_key
            latest_nonempty["fallback_reason"] = "interactive_cached_snapshot"
            return jsonify(latest_nonempty), 200
    result = _scan_daily_unusual_options(
        limit=int(body.get("limit", 50)),
        universe_size=int(body.get("universe_size", UNIVERSE_SCAN_LIMIT)),
        refresh=bool(body.get("refresh", False)),
        min_dte=int(body.get("min_dte", 1)),
        max_dte=int(body.get("max_dte", 45)),
        min_volume=int(body.get("min_volume", 200)),
        min_premium=float(body.get("min_premium", 100000)),
        max_symbols=int(body.get("max_symbols", UNUSUAL_SCAN_SYMBOL_LIMIT)),
        top_per_symbol=int(body.get("top_per_symbol", 3)),
        use_library_bias=bool(body.get("use_library_bias", True)),
        cache_max_age_hours=float(body.get("cache_max_age_hours", 4)),
        prefilter_multiplier=float(body.get("prefilter_multiplier", UNUSUAL_PREFILTER_MULTIPLIER)),
        max_runtime_seconds=float(body.get("max_runtime_seconds", 35)),
    )
    code = 200 if "error" not in result else 400
    return jsonify(result), code


@app.route("/api/learning/status", methods=["GET"])
def learning_status():
    return jsonify(_adaptive_learning_profile())


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)



