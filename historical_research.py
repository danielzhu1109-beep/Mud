from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import logging
import math
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

MIN_TRAIN_DAYS = 252
DEFAULT_THRESHOLD = 1.15
DEFAULT_HORIZON_DAYS = 10
DEFAULT_FORECAST_HORIZONS = [3, 5, 7, 10]
OPTIONDATA_BASE_URL = "https://optiondata.org"
OPTIONDATA_FREE_2013_MONTHS = ["2013-01", "2013-02", "2013-03", "2013-04", "2013-05", "2013-06"]

STOCK_FEATURE_COLUMNS = [
    "ret_5",
    "ret_20",
    "sma20_gap",
    "sma50_gap",
    "sma200_gap",
    "rsi_centered",
    "drawdown_63",
    "drawdown_252",
    "bounce_20",
    "volume_ratio20",
    "hv20",
    "hv60",
    "atr14_pct",
]

OPTION_FEATURE_COLUMNS = [
    "cp_volume_bias",
    "cp_oi_bias",
    "net_premium_bias",
    "atm_iv",
    "pc_skew_iv",
    "iv_hv_spread",
    "option_activity_ratio",
]


def _safe(value: Any, digits: int = 2) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
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


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def default_state() -> dict[str, Any]:
    return {
        "updated_at": None,
        "years": 5,
        "horizon_days": DEFAULT_HORIZON_DAYS,
        "threshold": DEFAULT_THRESHOLD,
        "mode": "empty",
        "summary": "historical research not initialized",
        "symbols": {},
        "aggregate": {},
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
        },
        "direction_weights": {"CALL": 0.0, "PUT": 0.0},
        "symbol_weights": {},
        "feature_weights": {},
        "coverage": {
            "stock_symbols": 0,
            "option_symbols": 0,
            "stock_rows": 0,
            "option_rows": 0,
            "option_source_files": [],
            "option_date_from": None,
            "option_date_to": None,
            "option_rows_last_5y": 0,
            "option_symbols_last_5y": 0,
            "option_coverage_quality": "none",
            "option_coverage_notes": [],
        },
    }


def _state_summary_blocks(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    symbols = state.get("symbols") or {}
    valid = [item for item in symbols.values() if isinstance(item, dict) and item.get("status") == "ok"]
    leaders = sorted(
        valid,
        key=lambda item: (
            _num(item.get("quality_score")),
            _num(item.get("research_confidence")),
            _num(item.get("trade_count")),
        ),
        reverse=True,
    )[:12]
    option_gaps = sorted(
        [item for item in valid if int(item.get("option_feature_trade_count", 0) or 0) <= 0],
        key=lambda item: (
            _num(item.get("trade_count")),
            _num(item.get("research_confidence")),
            _num(item.get("quality_score")),
        ),
        reverse=True,
    )[:12]

    coverage = state.get("coverage") or {}
    aggregate = state.get("aggregate") or {}
    next_actions: list[str] = []
    option_quality = str(coverage.get("option_coverage_quality") or "none").lower()
    if int(coverage.get("option_rows", 0) or 0) <= 0 and option_gaps:
        focus = ", ".join(str(item.get("symbol") or "") for item in option_gaps[:5] if item.get("symbol"))
        next_actions.append(f"import 5Y option history for: {focus}")
    elif option_quality in {"sample", "stale", "partial"}:
        notes = coverage.get("option_coverage_notes") or []
        if notes:
            next_actions.append(f"upgrade option history coverage: {notes[0]}")
        else:
            next_actions.append("upgrade option history coverage toward a recent multi-year dataset")
    if int(aggregate.get("symbol_count", 0) or 0) < 6:
        next_actions.append("expand the research symbol pool to improve cross-symbol stability")
    if float(aggregate.get("research_confidence", 0.0) or 0.0) < 0.45:
        next_actions.append("increase sample depth or extend horizon tuning to raise research confidence")
    return leaders, option_gaps, next_actions


def _upgrade_state(payload: dict[str, Any]) -> dict[str, Any]:
    base = default_state()
    upgraded = dict(base)
    upgraded.update(payload or {})
    for key in ("factor_weights", "direction_weights", "symbol_weights", "feature_weights", "coverage", "aggregate"):
        merged = dict(base.get(key, {}) or {})
        merged.update((payload or {}).get(key, {}) or {})
        upgraded[key] = merged

    symbols = upgraded.get("symbols") or {}
    normalized_symbols: dict[str, dict[str, Any]] = {}
    for raw_symbol, item in symbols.items():
        if not isinstance(item, dict):
            continue
        symbol = _normalize_symbol(item.get("symbol") or raw_symbol)
        current = dict(item)
        current["symbol"] = symbol
        if current.get("status") == "ok":
            if current.get("research_confidence") is None:
                current["research_confidence"] = _research_confidence(current)
            if current.get("quality_score") is None:
                current["quality_score"] = _quality_score(current)
        else:
            current["research_confidence"] = _num(current.get("research_confidence"))
            current["quality_score"] = _num(current.get("quality_score"))
        normalized_symbols[symbol] = current
    upgraded["symbols"] = normalized_symbols

    aggregate = upgraded.get("aggregate") or {}
    if normalized_symbols:
        valid = [item for item in normalized_symbols.values() if item.get("status") == "ok"]
        if valid:
            if aggregate.get("research_confidence") is None:
                aggregate["research_confidence"] = round(sum(_num(item.get("research_confidence")) for item in valid) / max(len(valid), 1), 3)
            if aggregate.get("option_feature_ratio") is None:
                aggregate["option_feature_ratio"] = round(
                    sum(min(1.0, _num(item.get("option_feature_trade_count")) / max(_num(item.get("trade_count"), 1.0), 1.0)) for item in valid)
                    / max(len(valid), 1),
                    3,
                )
            aggregate.setdefault("symbol_count", len(valid))
            aggregate.setdefault("trade_count", int(sum(_num(item.get("trade_count")) for item in valid)))
    upgraded["aggregate"] = aggregate

    leaders, option_gaps, next_actions = _state_summary_blocks(upgraded)
    upgraded["leaders"] = [
        {
            "symbol": item.get("symbol"),
            "quality_score": item.get("quality_score"),
            "research_confidence": item.get("research_confidence"),
            "trade_count": item.get("trade_count"),
            "win_rate": item.get("win_rate"),
            "calmar": item.get("calmar"),
            "mode": item.get("mode"),
        }
        for item in leaders
    ]
    upgraded["option_data_gaps"] = [
        {
            "symbol": item.get("symbol"),
            "trade_count": item.get("trade_count"),
            "research_confidence": item.get("research_confidence"),
            "quality_score": item.get("quality_score"),
            "win_rate": item.get("win_rate"),
        }
        for item in option_gaps
    ]
    upgraded["next_actions"] = next_actions
    return upgraded


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return default_state()
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default_state()
        return _upgrade_state(payload)
    except Exception as exc:
        logger.warning("historical research state load failed: %s", exc)
        return default_state()


def save_state(state_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = _upgrade_state(dict(payload or {}))
    payload["updated_at"] = payload.get("updated_at") or _utc_now_iso()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "root": base_dir,
        "stock": base_dir / "stock",
        "options_raw": base_dir / "options_raw",
        "options_daily": base_dir / "options_daily",
        "datasets": base_dir / "datasets",
        "backtests": base_dir / "backtests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=True).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=True).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return np.where(df["Close"].replace(0, np.nan).notna(), atr / df["Close"] * 100, np.nan)


def _bucket_from_ivrv(value: Any) -> str:
    num = _num(value, 0.0)
    if num <= -5:
        return "cheap"
    if num >= 15:
        return "rich"
    return "neutral"


def _bucket_from_skew(value: Any) -> str:
    num = _num(value, 0.0)
    if num >= 2.5:
        return "supportive"
    if num <= -2.5:
        return "adverse"
    return "neutral"


def _bucket_from_flow(value: Any) -> str:
    num = _num(value, 1.0)
    if num >= 1.35:
        return "strong"
    if num <= 0.75:
        return "weak"
    return "normal"


def _merge_named_series(data: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in data.columns:
            return pd.to_numeric(data[name], errors="coerce")
    return pd.Series(default, index=data.index, dtype=float)


def _fetch_stock_history(symbol: str, years: int = 5) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{years}y", interval="1d", auto_adjust=False)
    if hist.empty:
        hist = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
    if hist.empty:
        return pd.DataFrame()
    hist = hist.copy()
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [col[0] for col in hist.columns]
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return hist


def backfill_stock_history(base_dir: Path, symbols: list[str], years: int = 5, refresh: bool = False) -> dict[str, Any]:
    dirs = ensure_dirs(base_dir)
    summary: dict[str, Any] = {"symbols": {}, "rows": 0}
    for raw_symbol in symbols:
        symbol = _normalize_symbol(raw_symbol)
        if not symbol:
            continue
        path = dirs["stock"] / f"{symbol}.csv"
        if path.exists() and not refresh:
            try:
                hist = load_stock_history(base_dir, symbol)
            except Exception:
                hist = pd.DataFrame()
        else:
            hist = _fetch_stock_history(symbol, years=years)
            if not hist.empty:
                hist.reset_index(names="Date").to_csv(path, index=False)
        if hist.empty:
            summary["symbols"][symbol] = {"status": "error", "rows": 0}
            continue
        summary["symbols"][symbol] = {
            "status": "ok",
            "rows": int(len(hist)),
            "from": hist.index.min().date().isoformat(),
            "to": hist.index.max().date().isoformat(),
            "path": str(path),
        }
        summary["rows"] += int(len(hist))
    return summary


def load_stock_history(base_dir: Path, symbol: str) -> pd.DataFrame:
    path = ensure_dirs(base_dir)["stock"] / f"{_normalize_symbol(symbol)}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Close"])


def _discover_option_files(base_dir: Path, option_source_paths: Optional[list[str]] = None) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()

    def _append(path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            return
        seen.add(key)
        files.append(path)

    if option_source_paths:
        for raw in option_source_paths:
            path = Path(str(raw).strip()).expanduser()
            if not path.exists():
                continue
            if path.is_dir():
                for pattern in ("*.csv", "*.json", "*.jsonl", "*.zip"):
                    for match in sorted(path.rglob(pattern)):
                        _append(match)
            elif path.suffix.lower() in {".csv", ".json", ".jsonl", ".zip"}:
                _append(path)
    else:
        option_dir = ensure_dirs(base_dir)["options_raw"]
        for pattern in ("*.csv", "*.json", "*.jsonl", "*.zip"):
            for match in sorted(option_dir.rglob(pattern)):
                _append(match)
    return files


def _csv_symbol_column(data: pd.DataFrame) -> Optional[str]:
    for name in ("symbol", "underlying", "underlying_symbol", "ticker", "root"):
        if name in data.columns:
            return name
    return None


def _read_csv_filtered(source: Any, symbol_filter: Optional[set[str]] = None) -> pd.DataFrame:
    if not symbol_filter:
        return pd.read_csv(source)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, chunksize=200000):
        symbol_col = _csv_symbol_column(chunk)
        if symbol_col:
            filtered = chunk[chunk[symbol_col].astype(str).str.strip().str.upper().isin(symbol_filter)].copy()
        else:
            filtered = chunk
        if not filtered.empty:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _zip_member_date(name: str) -> str:
    stem = Path(name).name
    for token in stem.replace("(", "_").replace(")", "_").replace(".", "_").split("_"):
        if len(token) == 10 and token[4] == "-" and token[7] == "-":
            return token
    return ""


def _read_option_zip(path: Path, symbol_filter: Optional[set[str]] = None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            return pd.DataFrame()
        stock_maps: dict[str, dict[str, float]] = {}
        for name in members:
            lower = name.lower()
            if "stock" not in lower:
                continue
            member_date = _zip_member_date(name)
            with archive.open(name) as handle:
                stock_df = _read_csv_filtered(handle, symbol_filter=symbol_filter)
            if stock_df.empty:
                continue
            stock_df.columns = [str(col).strip() for col in stock_df.columns]
            symbol_col = _csv_symbol_column(stock_df)
            if not symbol_col:
                continue
            close_series = _merge_named_series(stock_df, ["close", "Close", "adj_close", "last"])
            mapping = {
                _normalize_symbol(sym): _num(close, np.nan)
                for sym, close in zip(stock_df[symbol_col].astype(str), close_series, strict=False)
                if _normalize_symbol(sym)
            }
            if mapping:
                stock_maps[member_date] = mapping

        for name in members:
            lower = name.lower()
            if "option" not in lower:
                continue
            member_date = _zip_member_date(name)
            with archive.open(name) as handle:
                option_df = _read_csv_filtered(handle, symbol_filter=symbol_filter)
            if option_df.empty:
                continue
            option_df.columns = [str(col).strip() for col in option_df.columns]
            if "underlying_price" not in option_df.columns:
                symbol_col = _csv_symbol_column(option_df)
                if symbol_col and member_date in stock_maps:
                    option_df["underlying_price"] = option_df[symbol_col].astype(str).map(
                        lambda sym: stock_maps[member_date].get(_normalize_symbol(sym), np.nan)
                    )
            rows.append(option_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _read_option_file(path: Path, symbol_filter: Optional[set[str]] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_filtered(path, symbol_filter=symbol_filter)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".zip":
        return _read_option_zip(path, symbol_filter=symbol_filter)
    return pd.DataFrame()


def _normalize_option_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("C"):
        return "CALL"
    if text.startswith("P"):
        return "PUT"
    return ""


def _first_present(data: pd.DataFrame, names: list[str], default: Any = None) -> pd.Series:
    for name in names:
        if name in data.columns:
            return data[name]
    return pd.Series(default, index=data.index)


def _normalize_option_history(raw: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    data = raw.copy()
    data.columns = [str(col).strip() for col in data.columns]
    normalized = pd.DataFrame(index=data.index)
    normalized["date"] = pd.to_datetime(
        _first_present(data, ["date", "trade_date", "quote_date", "as_of_date", "timestamp", "datetime"]),
        errors="coerce",
    ).dt.normalize()
    normalized["symbol"] = _first_present(data, ["symbol", "underlying", "underlying_symbol", "ticker", "root"], "").map(_normalize_symbol)
    normalized["option_type"] = _first_present(data, ["option_type", "type", "right", "call_put", "cp_flag"], "").map(_normalize_option_type)
    normalized["expiry"] = pd.to_datetime(
        _first_present(data, ["expiry", "expiration", "expiration_date", "exp_date", "maturity"]),
        errors="coerce",
    ).dt.normalize()
    normalized["strike"] = pd.to_numeric(
        _first_present(data, ["strike", "strike_price", "exercise_price", "strikePrice"]),
        errors="coerce",
    )
    normalized["underlying_price"] = pd.to_numeric(
        _first_present(data, ["underlying_price", "spot", "spot_price", "stock_price", "underlier_price", "close_underlying"]),
        errors="coerce",
    )
    normalized["bid"] = pd.to_numeric(_first_present(data, ["bid", "bid_price"]), errors="coerce")
    normalized["ask"] = pd.to_numeric(_first_present(data, ["ask", "ask_price"]), errors="coerce")
    normalized["last"] = pd.to_numeric(_first_present(data, ["last", "last_price", "lastPrice"]), errors="coerce")
    normalized["mark"] = pd.to_numeric(_first_present(data, ["mark", "mid", "close", "settlement", "price"]), errors="coerce")
    normalized["volume"] = pd.to_numeric(_first_present(data, ["volume", "trade_volume", "option_volume"]), errors="coerce")
    normalized["oi"] = pd.to_numeric(_first_present(data, ["oi", "open_interest", "openInterest"]), errors="coerce")
    iv_raw = pd.to_numeric(_first_present(data, ["iv_pct", "implied_volatility", "impliedVolatility", "iv", "mark_iv"]), errors="coerce")
    normalized["iv_pct"] = np.where(iv_raw.abs() <= 3, iv_raw * 100, iv_raw)
    normalized["source_file"] = source_file

    normalized["mid"] = np.where(
        normalized["bid"].fillna(0) > 0,
        (normalized["bid"].fillna(0) + normalized["ask"].fillna(0)) / 2.0,
        np.nan,
    )
    normalized["close"] = normalized["mark"].where(normalized["mark"].notna(), normalized["last"])
    normalized["close"] = normalized["close"].where(normalized["close"].notna(), normalized["mid"])
    normalized["premium"] = pd.to_numeric(_first_present(data, ["premium", "notional_premium"]), errors="coerce")
    normalized["premium"] = normalized["premium"].where(
        normalized["premium"].notna(),
        normalized["close"].fillna(0).clip(lower=0) * normalized["volume"].fillna(0).clip(lower=0) * 100,
    )

    normalized = normalized.dropna(subset=["date", "symbol"])
    normalized = normalized[normalized["option_type"].isin(["CALL", "PUT"])]
    normalized = normalized[normalized["strike"].notna() | normalized["volume"].notna() | normalized["oi"].notna()]
    return normalized.reset_index(drop=True)


def option_source_catalog() -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = [
        {
            "code": "alpha_vantage_historical_options",
            "type": "official_api",
            "coverage": "up to 15+ years by contract date, subject to Alpha Vantage plan",
            "requires": ["ALPHAVANTAGE_API_KEY", "premium_plan"],
            "notes": "Official historical options endpoint. This environment key exists, but the endpoint may still be locked without a premium subscription.",
            "url": "https://www.alphavantage.co/documentation/",
        },
        {
            "code": "optiondata_sample_2022_08_24",
            "type": "public_sample_zip",
            "coverage": "1 trading day snapshot with option and stock files",
            "requires": [],
            "notes": "Free public sample from optiondata.org. Useful to validate the pipeline and enrich a recent day inside the 5Y window.",
            "url": f"{OPTIONDATA_BASE_URL}/Sample2022-08-24.zip",
        },
        {
            "code": "optiondata_free_2013",
            "type": "public_archive_zip",
            "coverage": "Jan 2013 to Jun 2013 monthly archives",
            "requires": [],
            "notes": "Free public historical archive. Good for importer validation, but outside the current 5Y learning window.",
            "url": OPTIONDATA_BASE_URL,
        },
        {
            "code": "remote_url",
            "type": "generic_download",
            "coverage": "any direct CSV/JSON/JSONL/ZIP URL, including GitHub raw and release assets",
            "requires": [],
            "notes": "Use this for GitHub datasets or vendor-hosted files when you have direct download URLs.",
            "url": "https://github.com/",
        },
    ]
    return sources


def _option_source_score(status: str, coverage_quality: str = "", source_type: str = "") -> int:
    status = str(status or "").lower()
    coverage_quality = str(coverage_quality or "").lower()
    source_type = str(source_type or "").lower()
    score = 0
    if status == "ok":
        score += 40
    elif status in {"cached", "available"}:
        score += 30
    elif status == "locked":
        score += 10
    elif status == "missing_api_key":
        score += 5
    quality_boost = {
        "recent_dense": 55,
        "recent_multi_day": 45,
        "partial": 22,
        "sample": 10,
        "stale": 5,
        "none": 0,
    }
    score += quality_boost.get(coverage_quality, 0)
    if source_type == "official_api":
        score += 12
    elif source_type == "generic_download":
        score += 7
    elif source_type == "public_archive_zip":
        score += 4
    elif source_type == "public_sample_zip":
        score += 2
    return max(0, min(100, score))


def _download_file(url: str, dest: Path, timeout: int = 120) -> dict[str, Any]:
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            fh.write(chunk)
            total += len(chunk)
    sha256 = _sha256_file(dest)
    return {"path": str(dest), "bytes": total, "url": url, "sha256": sha256}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_remote_filename(url: str, index: int = 1) -> str:
    parsed = urlparse(url)
    base = Path(parsed.path).name or f"remote_{index}.dat"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    stem = Path(base).stem or f"remote_{index}"
    suffix = Path(base).suffix or ".dat"
    return f"{stem}_{digest}{suffix}"


def _is_supported_remote_file(url: str) -> bool:
    lower = str(url or "").lower()
    return any(lower.endswith(ext) for ext in (".csv", ".json", ".jsonl", ".zip"))


def _github_blob_to_raw(url: str) -> str:
    match = re.match(r"^https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", url)
    if not match:
        return url
    owner, repo, ref, path = match.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"


def _github_release_assets(url: str, timeout: int = 45) -> list[str]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []
    matches = re.findall(r'href="([^"]+/releases/download/[^"]+\.(?:zip|csv|json|jsonl))"', html, flags=re.I)
    assets: list[str] = []
    for href in matches:
        full = href if href.startswith("http") else f"https://github.com{href}"
        if full not in assets:
            assets.append(full)
    return assets


def _expand_remote_urls(remote_urls: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    expanded: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_url in remote_urls:
        url = str(raw_url or "").strip()
        if not url:
            continue
        resolved: list[str] = []
        note = ""
        if "/blob/" in url and "github.com" in url:
            resolved = [_github_blob_to_raw(url)]
            note = "github_blob_resolved_to_raw"
        elif "raw.githubusercontent.com" in url or _is_supported_remote_file(url):
            resolved = [_github_blob_to_raw(url) if "/blob/" in url else url]
            if "/blob/" in url:
                note = "github_blob_resolved_to_raw"
        elif "github.com" in url and ("/releases" in url or url.rstrip("/").count("/") <= 4):
            resolved = _github_release_assets(url)
            note = "github_release_assets_discovered" if resolved else "github_page_no_supported_assets_found"
        else:
            resolved = [url]

        unique_resolved: list[str] = []
        for item in resolved:
            if item and item not in seen:
                seen.add(item)
                expanded.append(item)
                unique_resolved.append(item)
        diagnostics.append(
            {
                "input_url": url,
                "resolved_urls": unique_resolved,
                "status": "ok" if unique_resolved else "skipped",
                "note": note or ("direct_file" if unique_resolved else "unsupported_or_empty"),
            }
        )
    return expanded, diagnostics


def _load_bootstrap_manifest(base_dir: Path) -> dict[str, Any]:
    return _load_json_if_exists(ensure_dirs(base_dir)["root"] / "option_bootstrap_manifest.json")


def _save_bootstrap_manifest(base_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    dirs = ensure_dirs(base_dir)
    manifest_path = dirs["root"] / "option_bootstrap_manifest.json"
    payload = dict(payload or {})
    payload["updated_at"] = _utc_now_iso()
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    return payload


def _merge_bootstrap_results(existing: list[dict[str, Any]], new_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*(existing or []), *(new_results or [])]:
        code = str(item.get("code") or "")
        url = str(item.get("url") or "")
        path = str(item.get("path") or "")
        key = (code, url, path)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[-200:]


def _alpha_vantage_probe(api_key: str) -> dict[str, Any]:
    if not api_key:
        return {"status": "missing_api_key"}
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "HISTORICAL_OPTIONS", "symbol": "AAPL", "date": "2024-01-19", "apikey": api_key},
            timeout=45,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    info = str((payload or {}).get("Information") or "")
    if "premium endpoint" in info.lower():
        return {"status": "locked", "message": info}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return {"status": "ok", "sample_contracts": len(payload.get("data", []))}
    return {"status": "unexpected", "message": str(payload)[:600]}


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def assess_option_sources(base_dir: Path, symbols: Optional[list[str]] = None) -> dict[str, Any]:
    dirs = ensure_dirs(base_dir)
    catalog = option_source_catalog()
    bootstrap_manifest = _load_json_if_exists(dirs["root"] / "option_bootstrap_manifest.json")
    options_manifest = _load_json_if_exists(dirs["options_daily"] / "manifest.json")
    local_quality = str(options_manifest.get("coverage_quality") or "none")
    local_notes = options_manifest.get("coverage_notes", []) if isinstance(options_manifest.get("coverage_notes"), list) else []
    source_results = bootstrap_manifest.get("results", []) if isinstance(bootstrap_manifest.get("results"), list) else []
    source_result_map: dict[str, Any] = {}
    for item in source_results:
        code = str(item.get("code") or "").strip()
        if code and code not in source_result_map:
            source_result_map[code] = item

    try:
        import os
        alpha_probe = _alpha_vantage_probe(os.getenv("ALPHAVANTAGE_API_KEY", "").strip())
    except Exception as exc:
        alpha_probe = {"status": "error", "error": str(exc)}

    assessed: list[dict[str, Any]] = []
    for source in catalog:
        item = dict(source)
        code = str(item.get("code") or "")
        source_type = str(item.get("type") or "")
        prior = dict(source_result_map.get(code) or {})
        status = str(prior.get("status") or "unknown")
        coverage_quality = "none"
        notes: list[str] = []
        if code == "alpha_vantage_historical_options":
            status = str(alpha_probe.get("status") or status or "unknown")
            if status == "locked":
                notes.append("official endpoint detected, but the current key does not unlock premium historical options")
            elif status == "ok":
                coverage_quality = "recent_multi_day"
                notes.append("official source is reachable and suitable for recent multi-year option history")
            elif status == "missing_api_key":
                notes.append("no Alpha Vantage key configured")
            elif alpha_probe.get("message"):
                notes.append(str(alpha_probe.get("message")))
        elif code == "optiondata_sample_2022_08_24":
            status = "cached" if (dirs["options_raw"] / "remote" / "optiondata" / "Sample2022-08-24.zip").exists() else status
            coverage_quality = "sample"
            notes.append("public sample is useful for importer validation, not for full recent learning")
        elif code == "optiondata_free_2013":
            month_paths = [(dirs["options_raw"] / "remote" / "optiondata" / f"{month}.zip") for month in OPTIONDATA_FREE_2013_MONTHS]
            status = "cached" if any(path.exists() for path in month_paths) else status
            coverage_quality = "stale"
            notes.append("free archive is outside the current 5-year learning window")
        elif code == "remote_url":
            custom_dir = dirs["options_raw"] / "remote" / "custom"
            status = "available" if custom_dir.exists() and any(custom_dir.iterdir()) else status
            coverage_quality = local_quality if local_quality != "none" else "none"
            if custom_dir.exists() and any(custom_dir.iterdir()):
                notes.append("custom remote files already downloaded; inspect coverage quality before relying on them")
            else:
                notes.append("best path for GitHub raw, release assets, and vendor-hosted direct files")

        if code != "alpha_vantage_historical_options" and not coverage_quality and local_quality != "none":
            coverage_quality = local_quality
        if code != "alpha_vantage_historical_options":
            notes.extend(str(note) for note in local_notes[:2] if str(note))

        item["status"] = status
        item["coverage_quality"] = coverage_quality or "none"
        item["score"] = _option_source_score(status=status, coverage_quality=item["coverage_quality"], source_type=source_type)
        item["priority"] = "high" if item["score"] >= 60 else "medium" if item["score"] >= 30 else "low"
        item["notes_live"] = notes[:4]
        if code in source_result_map:
            item["last_result"] = prior
        assessed.append(item)

    assessed.sort(key=lambda item: (int(item.get("score", 0)), str(item.get("code"))), reverse=True)
    best = assessed[0] if assessed else {}
    actions: list[str] = []
    if best:
        actions.append(f"highest-value source right now: {best.get('code')} ({best.get('priority')})")
    if str(alpha_probe.get("status") or "") == "locked":
        actions.append("Alpha Vantage historical options is officially supported but still locked behind a premium plan")
    if local_quality in {"sample", "partial", "stale", "none"}:
        actions.append("continue importing direct remote files or paid exports until recent multi-day coverage is reached")

    return {
        "status": "ok",
        "symbols": [_normalize_symbol(symbol) for symbol in (symbols or []) if _normalize_symbol(symbol)],
        "local_option_manifest": options_manifest,
        "alpha_vantage_probe": alpha_probe,
        "sources": assessed,
        "recommended_actions": actions,
    }


def option_source_inventory(base_dir: Path) -> dict[str, Any]:
    dirs = ensure_dirs(base_dir)
    manifest = _load_bootstrap_manifest(base_dir)
    rows: list[dict[str, Any]] = []
    duplicate_hashes: dict[str, list[str]] = {}
    changed = False
    for item in manifest.get("results", []) if isinstance(manifest, dict) else []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        path = str(item.get("path") or "")
        url = str(item.get("url") or "")
        sha256 = str(item.get("sha256") or "")
        exists = Path(path).exists() if path else False
        if exists and not sha256:
            sha256 = _sha256_file(Path(path))
            item["sha256"] = sha256
            changed = True
        size_bytes = Path(path).stat().st_size if exists else int(item.get("bytes", 0) or 0)
        row = {
            "code": code,
            "status": item.get("status"),
            "path": path,
            "url": url,
            "sha256": sha256,
            "exists": exists,
            "size_bytes": size_bytes,
            "downloaded": bool(item.get("downloaded")),
            "updated_at": manifest.get("updated_at"),
        }
        rows.append(row)
        if sha256:
            duplicate_hashes.setdefault(sha256, []).append(path or url or code)
    if changed and isinstance(manifest, dict):
        _save_bootstrap_manifest(base_dir, manifest)
    duplicates = [{"sha256": key, "items": value} for key, value in duplicate_hashes.items() if len(value) > 1]
    rows.sort(key=lambda row: (str(row.get("code")), str(row.get("path"))))
    return {
        "status": "ok",
        "root": str(dirs["options_raw"] / "remote"),
        "count": len(rows),
        "rows": rows,
        "duplicates": duplicates,
        "requested_urls": manifest.get("requested_urls", []) if isinstance(manifest, dict) else [],
        "resolved_urls": manifest.get("resolved_urls", []) if isinstance(manifest, dict) else [],
        "url_resolution": manifest.get("url_resolution", []) if isinstance(manifest, dict) else [],
    }


def bootstrap_option_sources(
    base_dir: Path,
    symbols: Optional[list[str]] = None,
    source_codes: Optional[list[str]] = None,
    remote_urls: Optional[list[str]] = None,
    refresh: bool = False,
) -> dict[str, Any]:
    dirs = ensure_dirs(base_dir)
    existing_manifest = _load_bootstrap_manifest(base_dir)
    symbols = [_normalize_symbol(symbol) for symbol in (symbols or []) if _normalize_symbol(symbol)]
    source_codes = [str(code).strip() for code in (source_codes or []) if str(code).strip()]
    input_remote_urls = [str(url).strip() for url in (remote_urls or []) if str(url).strip()]
    remote_urls, url_resolution = _expand_remote_urls(input_remote_urls)
    results: list[dict[str, Any]] = []

    if "alpha_vantage_historical_options" in source_codes:
        try:
            import os

            probe = _alpha_vantage_probe(os.getenv("ALPHAVANTAGE_API_KEY", "").strip())
        except Exception as exc:
            probe = {"status": "error", "error": str(exc)}
        probe["code"] = "alpha_vantage_historical_options"
        probe["downloaded"] = False
        results.append(probe)

    for code in source_codes:
        if code == "optiondata_sample_2022_08_24":
            url = f"{OPTIONDATA_BASE_URL}/Sample2022-08-24.zip"
            dest = dirs["options_raw"] / "remote" / "optiondata" / "Sample2022-08-24.zip"
            if dest.exists() and not refresh:
                results.append({"code": code, "status": "cached", "path": str(dest), "downloaded": False, "url": url, "sha256": _sha256_file(dest)})
                continue
            try:
                meta = _download_file(url, dest)
                results.append({"code": code, "status": "ok", "downloaded": True, **meta})
            except Exception as exc:
                results.append({"code": code, "status": "error", "error": str(exc), "url": url})
        elif code == "optiondata_free_2013":
            months = OPTIONDATA_FREE_2013_MONTHS
            month_results: list[dict[str, Any]] = []
            for month in months:
                url = f"{OPTIONDATA_BASE_URL}/{month}.zip"
                dest = dirs["options_raw"] / "remote" / "optiondata" / f"{month}.zip"
                if dest.exists() and not refresh:
                    month_results.append({"month": month, "status": "cached", "path": str(dest), "downloaded": False, "url": url, "sha256": _sha256_file(dest)})
                    continue
                try:
                    meta = _download_file(url, dest)
                    month_results.append({"month": month, "status": "ok", "downloaded": True, **meta})
                except Exception as exc:
                    month_results.append({"month": month, "status": "error", "error": str(exc), "url": url})
            results.append({"code": code, "status": "ok", "months": month_results, "downloaded": any(item.get("downloaded") for item in month_results)})

    for idx, url in enumerate(remote_urls, start=1):
        filename = _safe_remote_filename(url, index=idx)
        dest = dirs["options_raw"] / "remote" / "custom" / filename
        if dest.exists() and not refresh:
            results.append({"code": "remote_url", "status": "cached", "path": str(dest), "url": url, "downloaded": False, "sha256": _sha256_file(dest)})
            continue
        try:
            meta = _download_file(url, dest)
            results.append({"code": "remote_url", "status": "ok", "downloaded": True, **meta})
        except Exception as exc:
            results.append({"code": "remote_url", "status": "error", "error": str(exc), "url": url})

    manifest = {
        "status": "ok",
        "symbols": symbols,
        "requested_sources": source_codes,
        "requested_urls": input_remote_urls,
        "resolved_urls": remote_urls,
        "url_resolution": url_resolution,
        "results": _merge_bootstrap_results(existing_manifest.get("results", []) if isinstance(existing_manifest, dict) else [], results),
    }
    if isinstance(existing_manifest, dict):
        historical_symbols = existing_manifest.get("symbols", []) if isinstance(existing_manifest.get("symbols"), list) else []
        manifest["symbols"] = sorted({*historical_symbols, *symbols})
        historical_sources = existing_manifest.get("requested_sources", []) if isinstance(existing_manifest.get("requested_sources"), list) else []
        historical_urls = existing_manifest.get("requested_urls", []) if isinstance(existing_manifest.get("requested_urls"), list) else []
        historical_resolved = existing_manifest.get("resolved_urls", []) if isinstance(existing_manifest.get("resolved_urls"), list) else []
        manifest["requested_sources"] = sorted({*historical_sources, *source_codes})
        manifest["requested_urls"] = sorted({*historical_urls, *input_remote_urls})
        manifest["resolved_urls"] = sorted({*historical_resolved, *remote_urls})
    return _save_bootstrap_manifest(base_dir, manifest)


def _option_coverage_snapshot(full: pd.DataFrame, symbols: Optional[list[str]] = None) -> dict[str, Any]:
    if full is None or full.empty or "date" not in full.columns:
        return {
            "date_from": None,
            "date_to": None,
            "rows_last_5y": 0,
            "symbols_last_5y": 0,
            "quality": "none",
            "notes": ["no option rows imported"],
        }
    work = full.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"])
    if work.empty:
        return {
            "date_from": None,
            "date_to": None,
            "rows_last_5y": 0,
            "symbols_last_5y": 0,
            "quality": "none",
            "notes": ["option rows could not be dated"],
        }
    if symbols:
        target = {_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)}
        if target:
            work = work[work["symbol"].astype(str).str.upper().isin(target)].copy()
    if work.empty:
        return {
            "date_from": None,
            "date_to": None,
            "rows_last_5y": 0,
            "symbols_last_5y": 0,
            "quality": "none",
            "notes": ["no rows matched the requested symbols"],
        }
    date_from = work["date"].min()
    date_to = work["date"].max()
    cutoff = pd.Timestamp(dt.datetime.now(dt.UTC).date()) - pd.DateOffset(years=5)
    recent = work[work["date"] >= cutoff].copy()
    rows_last_5y = int(len(recent))
    symbols_last_5y = int(recent["symbol"].nunique()) if not recent.empty else 0
    date_span_days = int((date_to - date_from).days) if pd.notna(date_from) and pd.notna(date_to) else 0
    trading_days = int(recent["date"].nunique()) if not recent.empty else 0
    notes: list[str] = []
    quality = "none"
    if rows_last_5y <= 0:
        quality = "stale"
        notes.append("no option rows fall inside the last 5 years")
    elif trading_days <= 3:
        quality = "sample"
        notes.append(f"only {trading_days} trading day(s) are available in the last 5 years")
    elif trading_days < 120 or symbols_last_5y < max(2, min(6, len(symbols or []))):
        quality = "partial"
        notes.append(f"recent option history is thin: {trading_days} trading days across {symbols_last_5y} symbols")
    else:
        quality = "recent_multi_day"
        notes.append(f"recent option history spans {trading_days} trading days across {symbols_last_5y} symbols")
    if date_span_days > 0 and date_span_days < 365:
        notes.append(f"overall imported option span is only {date_span_days} calendar days")
    return {
        "date_from": date_from.date().isoformat() if pd.notna(date_from) else None,
        "date_to": date_to.date().isoformat() if pd.notna(date_to) else None,
        "rows_last_5y": rows_last_5y,
        "symbols_last_5y": symbols_last_5y,
        "trading_days_last_5y": trading_days,
        "date_span_days": date_span_days,
        "quality": quality,
        "notes": notes,
    }


def _dedupe_option_rows(full: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if full is None or full.empty:
        return pd.DataFrame(), {"rows_before": 0, "rows_after": 0, "duplicate_rows_removed": 0}
    work = full.copy()
    dedupe_keys = [
        "date",
        "symbol",
        "option_type",
        "expiry",
        "strike",
        "bid",
        "ask",
        "last",
        "mark",
        "volume",
        "oi",
        "iv_pct",
    ]
    for col in dedupe_keys:
        if col not in work.columns:
            work[col] = np.nan
    completeness_cols = [col for col in ("underlying_price", "bid", "ask", "last", "mark", "volume", "oi", "iv_pct", "expiry", "strike") if col in work.columns]
    work["_completeness"] = work[completeness_cols].notna().sum(axis=1)
    rows_before = int(len(work))
    work = work.sort_values(["_completeness", "source_file"], ascending=[False, True]).drop_duplicates(subset=dedupe_keys, keep="first")
    rows_after = int(len(work))
    work = work.drop(columns=["_completeness"], errors="ignore").reset_index(drop=True)
    return work, {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicate_rows_removed": max(0, rows_before - rows_after),
    }


def import_option_history(
    base_dir: Path,
    option_source_paths: Optional[list[str]] = None,
    refresh: bool = False,
    symbols: Optional[list[str]] = None,
) -> dict[str, Any]:
    dirs = ensure_dirs(base_dir)
    previous_manifest = _load_json_if_exists(dirs["options_daily"] / "manifest.json")
    files = _discover_option_files(base_dir, option_source_paths=option_source_paths)
    symbol_filter = {_normalize_symbol(symbol) for symbol in (symbols or []) if _normalize_symbol(symbol)}
    stock_cache: dict[str, pd.DataFrame] = {}
    imported_frames: list[pd.DataFrame] = []
    imported_files: list[str] = []
    for file_path in files:
        try:
            raw = _read_option_file(file_path, symbol_filter=symbol_filter or None)
        except Exception as exc:
            logger.warning("option history read failed for %s: %s", file_path, exc)
            continue
        normalized = _normalize_option_history(raw, source_file=file_path.name)
        if normalized.empty:
            continue
        imported_frames.append(normalized)
        imported_files.append(str(file_path))
    if not imported_frames:
        return {
            "files": [],
            "symbols": {},
            "rows": 0,
            "status": "empty",
            "date_from": None,
            "date_to": None,
            "rows_last_5y": 0,
            "symbols_last_5y": 0,
            "coverage_quality": "none",
            "coverage_notes": ["no option source files were imported"],
            "dedupe": {"rows_before": 0, "rows_after": 0, "duplicate_rows_removed": 0},
        }

    full = pd.concat(imported_frames, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"], errors="coerce").dt.normalize()
    full = full.dropna(subset=["date", "symbol"])
    full, dedupe_stats = _dedupe_option_rows(full)
    coverage_snapshot = _option_coverage_snapshot(full, symbols=symbols)
    for symbol in sorted(full["symbol"].dropna().unique().tolist()):
        stock_cache[symbol] = load_stock_history(base_dir, symbol)

    work = full.copy()
    missing_spot = work["underlying_price"].isna()
    if missing_spot.any():
        close_map: dict[tuple[str, str], float] = {}
        for symbol, hist in stock_cache.items():
            if hist.empty:
                continue
            tmp = hist.reset_index()
            for _, row in tmp.iterrows():
                close_map[(symbol, row["Date"].date().isoformat())] = _num(row["Close"], np.nan)
        work.loc[missing_spot, "underlying_price"] = [
            close_map.get((row["symbol"], row["date"].date().isoformat()), np.nan)
            for _, row in work.loc[missing_spot, ["symbol", "date"]].iterrows()
        ]

    work["dte"] = (pd.to_datetime(work["expiry"], errors="coerce") - pd.to_datetime(work["date"], errors="coerce")).dt.days
    work["atm_distance"] = (work["strike"] - work["underlying_price"]).abs()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
    work["oi"] = pd.to_numeric(work["oi"], errors="coerce").fillna(0)
    work["premium"] = pd.to_numeric(work["premium"], errors="coerce").fillna(0)
    work["iv_pct"] = pd.to_numeric(work["iv_pct"], errors="coerce")

    grouped = work.groupby(["date", "symbol", "option_type"], dropna=False).agg(
        total_volume=("volume", "sum"),
        total_oi=("oi", "sum"),
        total_premium=("premium", "sum"),
        mean_iv_pct=("iv_pct", "mean"),
        underlying_price=("underlying_price", "mean"),
    ).reset_index()

    pivot = grouped.pivot_table(
        index=["date", "symbol"],
        columns="option_type",
        values=["total_volume", "total_oi", "total_premium", "mean_iv_pct"],
        aggfunc="first",
    )
    pivot.columns = [f"{a.lower()}_{b.lower()}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    atm_rows = work[(work["dte"].fillna(9999) >= 7) & (work["dte"].fillna(9999) <= 60)].copy()
    atm_rows = atm_rows.dropna(subset=["atm_distance", "iv_pct"])
    atm_selected = pd.DataFrame()
    if not atm_rows.empty:
        atm_selected = (
            atm_rows.sort_values(["date", "symbol", "option_type", "dte", "atm_distance"])
            .groupby(["date", "symbol", "option_type"], as_index=False)
            .first()
        )
        atm_selected = atm_selected.pivot_table(
            index=["date", "symbol"],
            columns="option_type",
            values=["iv_pct", "dte"],
            aggfunc="first",
        )
        atm_selected.columns = [f"atm_{a.lower()}_{b.lower()}" for a, b in atm_selected.columns]
        atm_selected = atm_selected.reset_index()

    daily = pivot.merge(atm_selected, on=["date", "symbol"], how="left") if not atm_selected.empty else pivot
    daily["call_volume"] = pd.to_numeric(daily.get("total_volume_call"), errors="coerce").fillna(0)
    daily["put_volume"] = pd.to_numeric(daily.get("total_volume_put"), errors="coerce").fillna(0)
    daily["call_oi"] = pd.to_numeric(daily.get("total_oi_call"), errors="coerce").fillna(0)
    daily["put_oi"] = pd.to_numeric(daily.get("total_oi_put"), errors="coerce").fillna(0)
    daily["call_premium"] = pd.to_numeric(daily.get("total_premium_call"), errors="coerce").fillna(0)
    daily["put_premium"] = pd.to_numeric(daily.get("total_premium_put"), errors="coerce").fillna(0)
    daily["call_iv"] = pd.to_numeric(daily.get("atm_iv_pct_call"), errors="coerce")
    daily["put_iv"] = pd.to_numeric(daily.get("atm_iv_pct_put"), errors="coerce")
    daily["atm_iv"] = daily[["call_iv", "put_iv"]].mean(axis=1)
    daily["pc_skew_iv"] = daily["put_iv"] - daily["call_iv"]
    daily["cp_volume_bias"] = np.where(
        (daily["call_volume"] + daily["put_volume"]) > 0,
        (daily["call_volume"] - daily["put_volume"]) / (daily["call_volume"] + daily["put_volume"]),
        0.0,
    )
    daily["cp_oi_bias"] = np.where(
        (daily["call_oi"] + daily["put_oi"]) > 0,
        (daily["call_oi"] - daily["put_oi"]) / (daily["call_oi"] + daily["put_oi"]),
        0.0,
    )
    daily["net_premium_bias"] = np.where(
        (daily["call_premium"] + daily["put_premium"]) > 0,
        (daily["call_premium"] - daily["put_premium"]) / (daily["call_premium"] + daily["put_premium"]),
        0.0,
    )
    daily["total_option_volume"] = daily["call_volume"] + daily["put_volume"]
    daily["total_option_oi"] = daily["call_oi"] + daily["put_oi"]
    daily["total_option_premium"] = daily["call_premium"] + daily["put_premium"]
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)

    summary_symbols: dict[str, Any] = {}
    total_rows = 0
    for symbol in sorted(daily["symbol"].dropna().unique().tolist()):
        item = daily[daily["symbol"] == symbol].copy()
        item = item.sort_values("date")
        item["option_activity_ratio"] = item["total_option_volume"] / item["total_option_volume"].rolling(20).mean().replace(0, np.nan)
        item["option_activity_ratio"] = item["option_activity_ratio"].replace([np.inf, -np.inf], np.nan)
        output_path = dirs["options_daily"] / f"{symbol}.csv"
        if refresh or not output_path.exists():
            item.to_csv(output_path, index=False)
        summary_symbols[symbol] = {
            "rows": int(len(item)),
            "path": str(output_path),
            "option_dates": int(item["date"].nunique()),
            "has_iv": bool(item["atm_iv"].notna().any()),
        }
        total_rows += int(len(item))

    manifest = {
        "status": "ok",
        "files": imported_files,
        "rows": total_rows,
        "symbols": summary_symbols,
        "date_from": coverage_snapshot.get("date_from"),
        "date_to": coverage_snapshot.get("date_to"),
        "rows_last_5y": int(coverage_snapshot.get("rows_last_5y", 0) or 0),
        "symbols_last_5y": int(coverage_snapshot.get("symbols_last_5y", 0) or 0),
        "trading_days_last_5y": int(coverage_snapshot.get("trading_days_last_5y", 0) or 0),
        "coverage_quality": str(coverage_snapshot.get("quality") or "none"),
        "coverage_notes": coverage_snapshot.get("notes", []) or [],
        "dedupe": {
            **dedupe_stats,
            "unique_contract_rows": int(dedupe_stats.get("rows_after", 0) or 0),
            "previous_unique_contract_rows": int(((previous_manifest.get("dedupe") or {}).get("unique_contract_rows", 0) if isinstance(previous_manifest, dict) else 0) or 0),
            "delta_unique_contract_rows": int(dedupe_stats.get("rows_after", 0) or 0)
            - int(((previous_manifest.get("dedupe") or {}).get("unique_contract_rows", 0) if isinstance(previous_manifest, dict) else 0) or 0),
        },
    }
    (dirs["options_daily"] / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_option_daily(base_dir: Path, symbol: str) -> pd.DataFrame:
    path = ensure_dirs(base_dir)["options_daily"] / f"{_normalize_symbol(symbol)}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for col in df.columns:
        if col not in {"date", "symbol"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def inspect_option_sources(
    base_dir: Path,
    option_source_paths: Optional[list[str]] = None,
    max_files: int = 30,
) -> dict[str, Any]:
    files = _discover_option_files(base_dir, option_source_paths=option_source_paths)
    inspected: list[dict[str, Any]] = []
    total_raw_rows = 0
    total_normalized_rows = 0
    symbols: set[str] = set()
    date_min = None
    date_max = None

    for path in files[: max(1, int(max_files or 30))]:
        try:
            raw = _read_option_file(path)
            normalized = _normalize_option_history(raw, source_file=path.name)
        except Exception as exc:
            inspected.append(
                {
                    "path": str(path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        total_raw_rows += int(len(raw))
        total_normalized_rows += int(len(normalized))
        file_symbols = sorted({str(item) for item in normalized.get("symbol", pd.Series(dtype=str)).dropna().unique().tolist() if str(item)})
        if file_symbols:
            symbols.update(file_symbols)
        file_date_from = None
        file_date_to = None
        if not normalized.empty and "date" in normalized.columns:
            file_dates = pd.to_datetime(normalized["date"], errors="coerce").dropna()
            if not file_dates.empty:
                file_date_from = file_dates.min().date().isoformat()
                file_date_to = file_dates.max().date().isoformat()
                date_min = file_date_from if date_min is None else min(date_min, file_date_from)
                date_max = file_date_to if date_max is None else max(date_max, file_date_to)
        inspected.append(
            {
                "path": str(path),
                "status": "ok" if not normalized.empty else "empty",
                "rows_raw": int(len(raw)),
                "rows_normalized": int(len(normalized)),
                "columns": [str(col) for col in list(raw.columns)[:40]],
                "symbol_count": len(file_symbols),
                "symbols": file_symbols[:12],
                "date_from": file_date_from,
                "date_to": file_date_to,
                "has_iv": bool("iv_pct" in normalized.columns and normalized["iv_pct"].notna().any()),
                "has_underlying_price": bool("underlying_price" in normalized.columns and normalized["underlying_price"].notna().any()),
                "has_expiry": bool("expiry" in normalized.columns and normalized["expiry"].notna().any()),
            }
        )

    result = {
        "status": "ok",
        "file_count": len(files),
        "inspected_count": len(inspected),
        "rows_raw": total_raw_rows,
        "rows_normalized": total_normalized_rows,
        "symbol_count": len(symbols),
        "symbols": sorted(symbols)[:24],
        "date_from": date_min,
        "date_to": date_max,
        "files": inspected,
    }
    inspection_path = ensure_dirs(base_dir)["root"] / "options_inspection.json"
    inspection_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["inspection_path"] = str(inspection_path)
    return result

def _prepare_feature_frame(
    base_dir: Path,
    symbol: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    keep_unlabeled: bool = False,
) -> pd.DataFrame:
    stock = load_stock_history(base_dir, symbol)
    if stock.empty or len(stock) < MIN_TRAIN_DAYS + horizon_days + 20:
        return pd.DataFrame()
    stock = stock.copy().sort_index()
    stock["ret_1"] = stock["Close"].pct_change() * 100
    stock["ret_5"] = stock["Close"].pct_change(5) * 100
    stock["ret_20"] = stock["Close"].pct_change(20) * 100
    stock["sma20_gap"] = (stock["Close"] / stock["Close"].rolling(20).mean() - 1) * 100
    stock["sma50_gap"] = (stock["Close"] / stock["Close"].rolling(50).mean() - 1) * 100
    stock["sma200_gap"] = (stock["Close"] / stock["Close"].rolling(200).mean() - 1) * 100
    stock["rsi14"] = _rsi(stock["Close"], 14)
    stock["rsi_centered"] = stock["rsi14"] - 50
    stock["hv20"] = stock["ret_1"].div(100).rolling(20).std() * math.sqrt(252) * 100
    stock["hv60"] = stock["ret_1"].div(100).rolling(60).std() * math.sqrt(252) * 100
    stock["volume_ratio20"] = stock["Volume"] / stock["Volume"].rolling(20).mean().replace(0, np.nan)
    stock["drawdown_63"] = (stock["Close"] / stock["Close"].rolling(63).max() - 1) * 100
    stock["drawdown_252"] = (stock["Close"] / stock["Close"].rolling(252).max() - 1) * 100
    stock["bounce_20"] = (stock["Close"] / stock["Close"].rolling(20).min() - 1) * 100
    stock["atr14_pct"] = _atr_pct(stock, 14)
    stock["symbol"] = _normalize_symbol(symbol)

    option_daily = load_option_daily(base_dir, symbol)
    if not option_daily.empty:
        option_daily = option_daily.copy()
        option_daily["date"] = pd.to_datetime(option_daily["date"], errors="coerce")
        option_daily = option_daily.dropna(subset=["date"]).set_index("date").sort_index()
        option_daily["iv_hv_spread"] = option_daily["atm_iv"] - stock["hv20"]
    else:
        option_daily = pd.DataFrame(index=stock.index)

    merged = stock.join(option_daily.drop(columns=[c for c in ["symbol"] if c in option_daily.columns]), how="left")
    for col in OPTION_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    close = merged["Close"].to_numpy(dtype=float)
    high = merged["High"].to_numpy(dtype=float)
    low = merged["Low"].to_numpy(dtype=float)
    future_return = np.full(len(merged), np.nan)
    future_mae = np.full(len(merged), np.nan)
    future_mfe = np.full(len(merged), np.nan)

    for idx in range(len(merged)):
        exit_idx = idx + horizon_days
        if exit_idx >= len(merged):
            continue
        entry = close[idx]
        if not np.isfinite(entry) or entry <= 0:
            continue
        window_high = high[idx + 1 : exit_idx + 1]
        window_low = low[idx + 1 : exit_idx + 1]
        future_return[idx] = (close[exit_idx] / entry - 1) * 100
        future_mae[idx] = (np.nanmin(window_low) / entry - 1) * 100 if len(window_low) else np.nan
        future_mfe[idx] = (np.nanmax(window_high) / entry - 1) * 100 if len(window_high) else np.nan

    merged["future_return_pct"] = future_return
    merged["future_mae_pct"] = future_mae
    merged["future_mfe_pct"] = future_mfe
    merged["has_option_features"] = merged[OPTION_FEATURE_COLUMNS].notna().any(axis=1)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    if not keep_unlabeled:
        merged = merged.dropna(subset=["future_return_pct"])
    merged.index.name = "date"
    return merged


def build_research_dataset(base_dir: Path, symbol: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    return _prepare_feature_frame(base_dir, symbol, horizon_days=horizon_days, keep_unlabeled=False)


def _score_bucket(bucket: dict[str, float]) -> float:
    count = float(bucket.get("count", 0.0) or 0.0)
    if count <= 0:
        return 0.0
    avg_return = float(bucket.get("return_sum", 0.0) or 0.0) / max(count, 1e-6)
    win_rate = float(bucket.get("wins", 0.0) or 0.0) / max(count, 1e-6)
    sample_boost = min(1.0, count / 18.0)
    score = (win_rate - 0.5) * 4.8 + max(-2.8, min(2.8, avg_return / 4.5))
    return round(score * sample_boost, 2)


def _update_bucket(bucket: dict[str, float], trade_return: float, win: bool, weight: float = 1.0) -> None:
    bucket["count"] = float(bucket.get("count", 0.0) or 0.0) + weight
    bucket["wins"] = float(bucket.get("wins", 0.0) or 0.0) + (weight if win else 0.0)
    bucket["return_sum"] = float(bucket.get("return_sum", 0.0) or 0.0) + trade_return * weight


def _trade_drawdown(direction: str, mae_pct: float, mfe_pct: float) -> tuple[float, float]:
    if direction == "CALL":
        adverse = _num(mae_pct, 0.0)
        favorable = _num(mfe_pct, 0.0)
        return adverse, favorable
    adverse = -_num(mfe_pct, 0.0)
    favorable = -_num(mae_pct, 0.0)
    return adverse, favorable


def _feature_weight(train: pd.DataFrame, col: str) -> float:
    sample = train[[col, "future_return_pct"]].dropna()
    if len(sample) < 80:
        return 0.0
    x = sample[col].astype(float)
    y = sample["future_return_pct"].astype(float)
    x = x.clip(x.quantile(0.02), x.quantile(0.98))
    corr = x.corr(y)
    if pd.isna(corr):
        return 0.0
    median = x.median()
    high = sample.loc[x >= median, "future_return_pct"].mean()
    low = sample.loc[x < median, "future_return_pct"].mean()
    edge = _num(high, 0.0) - _num(low, 0.0)
    sample_boost = min(1.0, len(sample) / 260.0)
    weight = corr * 7.0 + edge * 0.12
    return round(max(-3.5, min(3.5, weight)) * sample_boost, 3)


def _zscore(train: pd.Series, value: float) -> float:
    train = pd.to_numeric(train, errors="coerce").dropna()
    if len(train) < 50:
        return 0.0
    std = float(train.std())
    if std <= 1e-9 or not np.isfinite(std):
        return 0.0
    mean = float(train.mean())
    z = (float(value) - mean) / std
    return max(-3.0, min(3.0, z))


def _score_current_row(
    train: pd.DataFrame,
    row: pd.Series,
    feature_weights: dict[str, float],
) -> tuple[float, list[dict[str, Any]], list[str]]:
    components: list[dict[str, Any]] = []
    used_features: list[str] = []
    score = 0.0
    for col, weight in feature_weights.items():
        if abs(_num(weight)) < 0.05 or col not in train.columns or pd.isna(row.get(col)):
            continue
        z = _zscore(train[col], row[col])
        contrib = float(weight) * z
        score += contrib
        used_features.append(col)
        components.append(
            {
                "feature": col,
                "weight": round(float(weight), 3),
                "value": _safe(row.get(col), 3),
                "z": round(z, 3),
                "contrib": round(contrib, 3),
            }
        )
    components.sort(key=lambda item: abs(_num(item.get("contrib"))), reverse=True)
    return round(score, 3), components[:10], used_features


def _option_signal_adjustment(
    latest: pd.Series,
    feature_weights: dict[str, float],
    option_trade_count: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    if not bool(latest.get("has_option_features")):
        return 0.0, []
    option_trade_scale = 0.8 + min(0.3, max(0, option_trade_count) / 120.0)
    drivers: list[dict[str, Any]] = []
    raw = 0.0
    rules = [
        ("cp_volume_bias", _num(latest.get("cp_volume_bias")), 1.15),
        ("cp_oi_bias", _num(latest.get("cp_oi_bias")), 0.9),
        ("net_premium_bias", _num(latest.get("net_premium_bias")), 1.2),
        ("pc_skew_iv", -_num(latest.get("pc_skew_iv")), 0.7),
        ("iv_hv_spread", -_num(latest.get("iv_hv_spread")) / 10.0, 0.55),
        ("option_activity_ratio", (_num(latest.get("option_activity_ratio")) - 1.0), 0.45),
    ]
    for feature, signal, base_scale in rules:
        if abs(signal) <= 1e-9:
            continue
        learned_weight = abs(_num(feature_weights.get(feature), 0.0))
        scale = base_scale * (0.85 + min(0.45, learned_weight * 0.08)) * option_trade_scale
        contrib = signal * scale
        raw += contrib
        drivers.append(
            {
                "feature": feature,
                "value": _safe(latest.get(feature), 3),
                "contrib": round(contrib, 3),
                "source": "option_context",
            }
        )
    drivers.sort(key=lambda item: abs(_num(item.get("contrib"))), reverse=True)
    return round(max(-2.4, min(2.4, raw)), 3), drivers[:6]


def _similarity_analogs(
    train: pd.DataFrame,
    row: pd.Series,
    feature_weights: dict[str, float],
    top_n: int = 24,
) -> pd.DataFrame:
    active = [col for col, weight in feature_weights.items() if abs(_num(weight)) >= 0.05 and col in train.columns and pd.notna(row.get(col))]
    if not active:
        return pd.DataFrame()
    sample = train.copy()
    distance = pd.Series(0.0, index=sample.index, dtype=float)
    used = 0
    for col in active:
        series = pd.to_numeric(sample[col], errors="coerce")
        std = float(series.std()) if len(series.dropna()) >= 30 else 0.0
        if std <= 1e-9 or not np.isfinite(std):
            continue
        weight = max(0.25, min(2.0, abs(_num(feature_weights.get(col)))))
        part = ((series - float(row[col])) / std).abs().fillna(9.0) * weight
        distance = distance.add(part, fill_value=0.0)
        used += 1
    if used <= 0:
        return pd.DataFrame()
    sample = sample.assign(_distance=distance / used)
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=["_distance", "future_return_pct"])
    if sample.empty:
        return sample
    return sample.nsmallest(max(8, min(top_n, len(sample))), "_distance").copy()


def _forecast_from_frame(
    state: dict[str, Any],
    symbol_state: dict[str, Any],
    symbol: str,
    feature_frame: pd.DataFrame,
    spot: Optional[float],
    horizon_days: int,
) -> dict[str, Any]:
    if feature_frame.empty or len(feature_frame) < MIN_TRAIN_DAYS + 10:
        return {
            "symbol": symbol,
            "status": "insufficient_data",
            "horizon_days": horizon_days,
        }

    labeled = feature_frame.dropna(subset=["future_return_pct"]).copy()
    if labeled.empty or len(labeled) < MIN_TRAIN_DAYS:
        return {
            "symbol": symbol,
            "status": "insufficient_labels",
            "horizon_days": horizon_days,
        }

    latest = feature_frame.iloc[-1]
    latest_ts = feature_frame.index[-1]
    train = labeled.copy()
    feature_weights = (symbol_state.get("feature_weights") or {}) if symbol_state else {}
    if not feature_weights:
        feature_cols = [col for col in STOCK_FEATURE_COLUMNS + OPTION_FEATURE_COLUMNS if col in train.columns]
        feature_weights = {col: _feature_weight(train, col) for col in feature_cols}
    score, components, used_features = _score_current_row(train, latest, feature_weights)
    option_trade_count = int(symbol_state.get("option_feature_trade_count", 0) or 0)
    option_adjustment, option_drivers = _option_signal_adjustment(latest, feature_weights, option_trade_count=option_trade_count)
    score += option_adjustment
    analogs = _similarity_analogs(train, latest, feature_weights, top_n=24)

    analog_return = 0.0
    analog_hit_rate = 50.0
    analog_mae = -2.0
    analog_mfe = 2.0
    analog_count = 0
    if not analogs.empty:
        analog_count = int(len(analogs))
        analogs["_similarity"] = 1.0 / (1.0 + analogs["_distance"].clip(lower=0.0))
        weight_sum = float(analogs["_similarity"].sum()) or 1.0
        analog_return = float((analogs["future_return_pct"] * analogs["_similarity"]).sum() / weight_sum)
        analog_mae = float((analogs["future_mae_pct"] * analogs["_similarity"]).sum() / weight_sum)
        analog_mfe = float((analogs["future_mfe_pct"] * analogs["_similarity"]).sum() / weight_sum)
        analog_hit_rate = float(((analogs["future_return_pct"] > 0).astype(float) * analogs["_similarity"]).sum() / weight_sum * 100)

    signal_strength = abs(score)
    direction = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
    if direction == "neutral" and abs(analog_return) >= 0.35:
        direction = "bullish" if analog_return > 0 else "bearish"
    expected_move_pct = analog_return * 0.55 + score * 0.18 + option_adjustment * 0.3
    if direction == "bearish" and expected_move_pct > 0:
        expected_move_pct *= -0.6
    elif direction == "bullish" and expected_move_pct < 0:
        expected_move_pct *= -0.6
    expected_move_pct = round(max(-12.0, min(12.0, expected_move_pct)), 2)

    if spot is None or not np.isfinite(float(spot or 0)) or float(spot or 0) <= 0:
        spot_value = float(latest.get("Close") or latest.get("underlying_price") or 0.0)
    else:
        spot_value = float(spot)
    if not np.isfinite(spot_value) or spot_value <= 0:
        return {
            "symbol": symbol,
            "status": "invalid_spot",
            "horizon_days": horizon_days,
        }

    expected_close = round(spot_value * (1 + expected_move_pct / 100.0), 2)
    band_low = round(spot_value * (1 + min(analog_mae, analog_return) / 100.0), 2)
    band_high = round(spot_value * (1 + max(analog_mfe, analog_return) / 100.0), 2)
    if band_low > band_high:
        band_low, band_high = band_high, band_low

    research_confidence = float(symbol_state.get("research_confidence", 0.0) or (state.get("aggregate") or {}).get("research_confidence", 0.0) or 0.0)
    quality_score = float(symbol_state.get("quality_score", 0.0) or 0.0)
    confidence = 0.38 + min(0.42, signal_strength / max(0.75, 1.0 + horizon_days * 0.05) * 0.08) + research_confidence * 0.28
    if analog_count >= 12:
        confidence += 0.04
    if option_adjustment and bool(latest.get("has_option_features")):
        confidence += 0.03
    confidence = round(max(0.12, min(0.92, confidence)), 3)

    probability_up = 50.0 + score * 4.2 + (analog_hit_rate - 50.0) * 0.45
    probability_up = max(8.0, min(92.0, probability_up))
    if direction == "bearish":
        probability_up = min(probability_up, 47.0)
    elif direction == "bullish":
        probability_up = max(probability_up, 53.0)
    probability_down = round(100.0 - probability_up, 2)
    probability_up = round(probability_up, 2)

    option_context = {
        "has_option_features": bool(latest.get("has_option_features")),
        "atm_iv": _safe(latest.get("atm_iv"), 2),
        "iv_hv_spread": _safe(latest.get("iv_hv_spread"), 2),
        "pc_skew_iv": _safe(latest.get("pc_skew_iv"), 2),
        "cp_volume_bias": _safe(latest.get("cp_volume_bias"), 3),
        "net_premium_bias": _safe(latest.get("net_premium_bias"), 3),
        "option_activity_ratio": _safe(latest.get("option_activity_ratio"), 2),
        "option_adjustment": round(option_adjustment, 3),
    }
    option_bias = "CALL" if direction == "bullish" else "PUT" if direction == "bearish" else "NEUTRAL"
    strategy = "directional_debit" if abs(expected_move_pct) >= 1.25 else "defined_risk_or_wait"
    if option_context["has_option_features"] and option_context.get("atm_iv") is not None and option_context["atm_iv"] >= 42:
        strategy = "prefer_spread_or_defined_risk"

    return {
        "symbol": symbol,
        "status": "ok",
        "forecast_date": latest_ts.strftime("%Y-%m-%d"),
        "target_date": (latest_ts + pd.tseries.offsets.BDay(horizon_days)).strftime("%Y-%m-%d"),
        "horizon_days": horizon_days,
        "spot": round(spot_value, 2),
        "direction": direction,
        "option_bias": option_bias,
        "strategy_bias": strategy,
        "expected_move_pct": expected_move_pct,
        "expected_close": expected_close,
        "range_low": band_low,
        "range_high": band_high,
        "confidence": confidence,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "signal_strength": round(signal_strength, 3),
        "score": round(score, 3),
        "analog_count": analog_count,
        "analog_expected_return_pct": round(analog_return, 2),
        "analog_avg_mae_pct": round(analog_mae, 2),
        "analog_avg_mfe_pct": round(analog_mfe, 2),
        "analog_hit_rate": round(analog_hit_rate, 2),
        "research_context": {
            "mode": symbol_state.get("mode", state.get("mode")),
            "research_confidence": round(research_confidence, 3),
            "quality_score": round(quality_score, 2),
            "bias": symbol_state.get("bias", (state.get("aggregate") or {}).get("direction_bias", "balanced")),
            "coverage_rows": int(symbol_state.get("coverage_rows", len(train)) or len(train)),
        },
        "option_context": option_context,
        "drivers": (components[:6] + option_drivers)[:8],
        "used_feature_count": len(used_features),
    }


def _evaluate_horizon_candidate(
    feature_frame: pd.DataFrame,
    horizon_days: int,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> dict[str, Any]:
    labeled = feature_frame.dropna(subset=["future_return_pct"]).copy()
    if labeled.empty or len(labeled) < min_train_days + 12:
        return {"horizon_days": horizon_days, "status": "insufficient_data"}
    test_size = max(18, min(48, len(labeled) // 5))
    start_idx = max(min_train_days, len(labeled) - test_size)
    feature_cols = [col for col in STOCK_FEATURE_COLUMNS + OPTION_FEATURE_COLUMNS if col in labeled.columns]
    wins = 0
    band_hits = 0
    errors: list[float] = []
    weighted_edges: list[float] = []
    coverage = 0
    evaluated = 0

    for idx in range(start_idx, len(labeled)):
        train = labeled.iloc[:idx].copy()
        if len(train) < min_train_days:
            continue
        row = labeled.iloc[idx]
        weights = {col: _feature_weight(train, col) for col in feature_cols}
        score, _, _ = _score_current_row(train, row, weights)
        option_adjustment, _ = _option_signal_adjustment(
            row,
            weights,
            option_trade_count=int(train["has_option_features"].sum()) if "has_option_features" in train.columns else 0,
        )
        score += option_adjustment
        analogs = _similarity_analogs(train, row, weights, top_n=16)
        analog_return = float(analogs["future_return_pct"].mean()) if not analogs.empty else 0.0
        expected_move = analog_return * 0.55 + score * 0.18 + option_adjustment * 0.3
        actual_move = _num(row.get("future_return_pct"))
        direction_ok = (expected_move > 0 and actual_move > 0) or (expected_move < 0 and actual_move < 0) or (abs(expected_move) <= 0.2 and abs(actual_move) <= 0.85)
        wins += 1 if direction_ok else 0
        entry = _num(row.get("Close"), 0.0)
        if entry > 0:
            expected_close = entry * (1 + expected_move / 100.0)
            actual_close = entry * (1 + actual_move / 100.0)
            errors.append(abs(actual_close - expected_close) / entry * 100.0)
            low = entry * (1 + min(_num(row.get("future_mae_pct")), analog_return) / 100.0)
            high = entry * (1 + max(_num(row.get("future_mfe_pct")), analog_return) / 100.0)
            if min(low, high) <= actual_close <= max(low, high):
                band_hits += 1
        weighted_edges.append(abs(expected_move))
        coverage += 1 if bool(row.get("has_option_features")) else 0
        evaluated += 1

    if evaluated <= 0:
        return {"horizon_days": horizon_days, "status": "no_evaluation"}
    direction_accuracy = wins / evaluated * 100.0
    band_hit_rate = band_hits / evaluated * 100.0
    avg_abs_error = sum(errors) / len(errors) if errors else 9.99
    avg_edge = sum(weighted_edges) / len(weighted_edges) if weighted_edges else 0.0
    option_ratio = coverage / evaluated
    score = (direction_accuracy - 50.0) * 0.07 + (band_hit_rate - 45.0) * 0.04 + avg_edge * 0.18 - avg_abs_error * 0.11 + option_ratio * 0.25
    return {
        "horizon_days": horizon_days,
        "status": "ok",
        "sample_size": evaluated,
        "direction_accuracy": round(direction_accuracy, 2),
        "band_hit_rate": round(band_hit_rate, 2),
        "avg_abs_error_pct": round(avg_abs_error, 2),
        "avg_expected_move_pct": round(avg_edge, 2),
        "option_feature_ratio": round(option_ratio, 3),
        "score": round(score, 3),
    }


def select_adaptive_horizon(
    base_dir: Path,
    symbol: str,
    candidate_horizons: Optional[list[int]] = None,
) -> dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    horizons = candidate_horizons or list(DEFAULT_FORECAST_HORIZONS)
    normalized_horizons: list[int] = []
    for horizon in horizons:
        value = max(3, min(int(horizon or DEFAULT_HORIZON_DAYS), 15))
        if value not in normalized_horizons:
            normalized_horizons.append(value)
    evaluations: list[dict[str, Any]] = []
    for horizon in normalized_horizons:
        frame = _prepare_feature_frame(base_dir, symbol, horizon_days=horizon, keep_unlabeled=True)
        evaluations.append(_evaluate_horizon_candidate(frame, horizon_days=horizon))
    valid = [item for item in evaluations if item.get("status") == "ok"]
    if not valid:
        fallback = normalized_horizons[0] if normalized_horizons else 5
        return {
            "symbol": symbol,
            "adaptive_enabled": True,
            "selected_horizon_days": fallback,
            "selection_reason": "fallback_no_valid_horizon",
            "candidates": evaluations,
        }
    valid.sort(
        key=lambda item: (
            _num(item.get("score")),
            _num(item.get("direction_accuracy")),
            -_num(item.get("avg_abs_error_pct"), 999.0),
            _num(item.get("sample_size")),
        ),
        reverse=True,
    )
    best = valid[0]
    return {
        "symbol": symbol,
        "adaptive_enabled": True,
        "selected_horizon_days": int(best.get("horizon_days") or 5),
        "selection_reason": "best_recent_walkforward_score",
        "selected_score": _safe(best.get("score"), 3),
        "candidates": evaluations,
    }


def default_forecast_memory() -> dict[str, Any]:
    return {
        "updated_at": None,
        "summary": {
            "total": 0,
            "resolved": 0,
            "pending": 0,
            "direction_accuracy": None,
            "band_hit_rate": None,
            "avg_abs_error_pct": None,
        },
        "forecasts": [],
    }


def load_forecast_memory(memory_path: Path) -> dict[str, Any]:
    if not memory_path.exists():
        return default_forecast_memory()
    try:
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default_forecast_memory()
        base = default_forecast_memory()
        base.update(payload)
        base["forecasts"] = payload.get("forecasts", []) if isinstance(payload.get("forecasts"), list) else []
        return base
    except Exception as exc:
        logger.warning("forecast memory load failed: %s", exc)
        return default_forecast_memory()


def save_forecast_memory(memory_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    payload["updated_at"] = _utc_now_iso()
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _equity_drawdown(points: list[float]) -> float:
    if not points:
        return 0.0
    series = pd.Series(points, dtype=float)
    peak = series.cummax().replace(0, np.nan)
    dd = series / peak - 1
    return round(float(dd.min()) * 100, 2) if not dd.empty else 0.0


def _research_confidence(result: dict[str, Any]) -> float:
    trade_count = max(0, int(result.get("trade_count", 0) or 0))
    if trade_count <= 0:
        return 0.0
    option_count = max(0, int(result.get("option_feature_trade_count", 0) or 0))
    sample_scale = min(1.0, trade_count / 120.0)
    option_ratio = option_count / max(trade_count, 1)
    option_scale = 0.82 + min(0.18, option_ratio * 0.18)
    drawdown = abs(min(0.0, _num(result.get("max_drawdown_pct"), 0.0)))
    drawdown_scale = 1.0 if drawdown <= 35 else max(0.58, 1.0 - (drawdown - 35) / 120.0)
    stability_scale = 0.85 + min(0.15, abs(_num(result.get("sharpe_like"), 0.0)) * 0.08)
    confidence = sample_scale * option_scale * drawdown_scale * stability_scale
    return round(max(0.05, min(1.0, confidence)), 3)


def _quality_score(result: dict[str, Any]) -> float:
    if result.get("status") != "ok":
        return 0.0
    win_edge = (_num(result.get("win_rate"), 50.0) - 50.0) * 0.16
    avg_edge = _num(result.get("avg_return_pct"), 0.0) * 0.7
    calmar_edge = _num(result.get("calmar"), 0.0) * 0.9
    drawdown = abs(min(0.0, _num(result.get("max_drawdown_pct"), 0.0)))
    drawdown_penalty = max(0.0, drawdown - 28.0) * 0.05
    option_boost = min(0.5, max(0.0, _num(result.get("option_feature_trade_count"), 0.0)) / max(1.0, _num(result.get("trade_count"), 1.0)) * 0.5)
    score = win_edge + avg_edge + calmar_edge - drawdown_penalty + option_boost
    return round(max(-6.0, min(6.0, score)), 2)


def _mode_from_option_manifest(option_manifest: dict[str, Any]) -> str:
    rows = int(option_manifest.get("rows", 0) or 0)
    quality = str(option_manifest.get("coverage_quality") or "none").lower()
    if rows <= 0:
        return "stock_only"
    if quality in {"recent_multi_day", "recent_dense", "recent"}:
        return "stock_plus_options"
    return "stock_plus_options_partial"


def backtest_symbol(
    dataset: pd.DataFrame,
    symbol: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    threshold: float = DEFAULT_THRESHOLD,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> dict[str, Any]:
    if dataset.empty or len(dataset) < min_train_days + horizon_days + 5:
        return {
            "symbol": symbol,
            "status": "insufficient_data",
            "trade_count": 0,
            "trades": [],
            "research_confidence": 0.0,
            "quality_score": 0.0,
        }

    feature_cols = [col for col in STOCK_FEATURE_COLUMNS + OPTION_FEATURE_COLUMNS if col in dataset.columns]
    factor_buckets = {
        "ivrv_cheap": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "ivrv_neutral": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "ivrv_rich": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "skew_supportive": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "skew_neutral": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "skew_adverse": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "flow_strong": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "flow_normal": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "flow_weak": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
    }
    direction_buckets = {
        "CALL": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
        "PUT": {"count": 0.0, "wins": 0.0, "return_sum": 0.0},
    }

    trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve = [{"ts": dataset.index[min_train_days - 1].strftime("%Y-%m-%d"), "equity": equity}]
    idx = min_train_days
    last_weights: dict[str, float] = {}

    while idx < len(dataset) - horizon_days:
        train = dataset.iloc[:idx].copy()
        row = dataset.iloc[idx]
        weights = {col: _feature_weight(train, col) for col in feature_cols}
        score = 0.0
        components: list[dict[str, Any]] = []
        for col, weight in weights.items():
            if abs(weight) < 0.02 or pd.isna(row.get(col)):
                continue
            z = _zscore(train[col], row[col])
            contrib = weight * z
            score += contrib
            components.append({"feature": col, "weight": round(weight, 3), "z": round(z, 3), "contrib": round(contrib, 3)})
        last_weights = weights
        if abs(score) < threshold:
            idx += 1
            continue

        direction = "CALL" if score > 0 else "PUT"
        realized = _num(row.get("future_return_pct"), 0.0)
        if direction == "PUT":
            realized = -realized
        adverse, favorable = _trade_drawdown(direction, _num(row.get("future_mae_pct"), 0.0), _num(row.get("future_mfe_pct"), 0.0))
        win = realized > 0
        equity *= max(0.01, 1 + realized / 100.0)
        exit_index = min(len(dataset) - 1, idx + horizon_days)
        exit_ts = dataset.index[exit_index]

        ivrv_bucket = _bucket_from_ivrv(row.get("iv_hv_spread"))
        skew_support = -_num(row.get("pc_skew_iv"), 0.0) if direction == "CALL" else _num(row.get("pc_skew_iv"), 0.0)
        skew_bucket = _bucket_from_skew(skew_support)
        flow_bucket = _bucket_from_flow(row.get("option_activity_ratio"))
        _update_bucket(factor_buckets[f"ivrv_{ivrv_bucket}"], realized, win)
        _update_bucket(factor_buckets[f"skew_{skew_bucket}"], realized, win)
        _update_bucket(factor_buckets[f"flow_{flow_bucket}"], realized, win)
        _update_bucket(direction_buckets[direction], realized, win)

        trade = {
            "symbol": symbol,
            "entry_date": dataset.index[idx].strftime("%Y-%m-%d"),
            "exit_date": exit_ts.strftime("%Y-%m-%d"),
            "direction": direction,
            "score": round(score, 3),
            "return_pct": round(realized, 2),
            "mae_pct": round(adverse, 2),
            "mfe_pct": round(favorable, 2),
            "win": win,
            "ivrv_bucket": ivrv_bucket,
            "skew_bucket": skew_bucket,
            "flow_bucket": flow_bucket,
            "has_option_features": bool(row.get("has_option_features")),
            "components": components[:8],
            "cp_volume_bias": _safe(row.get("cp_volume_bias"), 3),
            "net_premium_bias": _safe(row.get("net_premium_bias"), 3),
            "pc_skew_iv": _safe(row.get("pc_skew_iv"), 3),
            "iv_hv_spread": _safe(row.get("iv_hv_spread"), 3),
        }
        trades.append(trade)
        equity_curve.append({"ts": exit_ts.strftime("%Y-%m-%d"), "equity": round(equity, 6)})
        idx += horizon_days

    if not trades:
        return {
            "symbol": symbol,
            "status": "no_signals",
            "trade_count": 0,
            "trades": [],
            "feature_weights": last_weights,
            "research_confidence": 0.0,
            "quality_score": 0.0,
        }

    returns = [float(item["return_pct"]) for item in trades]
    mae_values = [float(item["mae_pct"]) for item in trades]
    mfe_values = [float(item["mfe_pct"]) for item in trades]
    call_count = sum(1 for item in trades if item["direction"] == "CALL")
    put_count = sum(1 for item in trades if item["direction"] == "PUT")
    total_return = (equity - 1) * 100
    max_drawdown = _equity_drawdown([float(item["equity"]) for item in equity_curve])
    avg_return = sum(returns) / len(returns)
    return_std = float(pd.Series(returns).std()) if len(returns) > 1 else 0.0
    sharpe_like = (avg_return / return_std) if return_std > 1e-9 else 0.0
    years_span = max(1 / 252.0, len(dataset) / 252.0)
    cagr = (equity ** (1 / years_span) - 1) * 100 if equity > 0 else -100.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else cagr
    direction_scores = {key: _score_bucket(bucket) for key, bucket in direction_buckets.items()}
    factor_scores = {key: _score_bucket(bucket) for key, bucket in factor_buckets.items()}
    bias = "balanced"
    if direction_scores["CALL"] - direction_scores["PUT"] >= 0.35:
        bias = "call"
    elif direction_scores["PUT"] - direction_scores["CALL"] >= 0.35:
        bias = "put"

    option_feature_trade_count = sum(1 for item in trades if item.get("has_option_features"))
    result = {
        "symbol": symbol,
        "status": "ok",
        "mode": "stock_plus_options" if option_feature_trade_count else "stock_only",
        "trade_count": len(trades),
        "call_count": call_count,
        "put_count": put_count,
        "win_rate": round(sum(1 for item in trades if item["win"]) / len(trades) * 100, 2),
        "avg_return_pct": round(avg_return, 2),
        "median_return_pct": round(float(pd.Series(returns).median()), 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "calmar": round(calmar, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_mae_pct": round(sum(mae_values) / len(mae_values), 2),
        "avg_mfe_pct": round(sum(mfe_values) / len(mfe_values), 2),
        "worst_trade_pct": round(min(returns), 2),
        "best_trade_pct": round(max(returns), 2),
        "sharpe_like": round(sharpe_like, 2),
        "direction_scores": direction_scores,
        "bias": bias,
        "feature_weights": {key: round(value, 3) for key, value in last_weights.items() if abs(_num(value)) >= 0.05},
        "factor_weights": factor_scores,
        "option_feature_trade_count": option_feature_trade_count,
        "coverage_rows": int(len(dataset)),
        "equity_curve": equity_curve[-260:],
        "trades": trades,
    }
    result["research_confidence"] = _research_confidence(result)
    result["quality_score"] = _quality_score(result)
    return result


def _aggregate_results(
    symbol_results: dict[str, dict[str, Any]],
    stock_summary: dict[str, Any],
    option_manifest: dict[str, Any],
    years: int,
    horizon_days: int,
    threshold: float,
) -> dict[str, Any]:
    valid = [item for item in symbol_results.values() if item.get("status") == "ok"]
    if not valid:
        payload = default_state()
        payload.update(
            {
                "updated_at": _utc_now_iso(),
                "years": years,
                "horizon_days": horizon_days,
                "threshold": threshold,
                "mode": _mode_from_option_manifest(option_manifest),
                "summary": "historical research finished, but no valid backtest signals were produced",
                "symbols": symbol_results,
                "coverage": {
                    "stock_symbols": len(stock_summary.get("symbols", {})),
                    "option_symbols": len(option_manifest.get("symbols", {})),
                    "stock_rows": int(stock_summary.get("rows", 0)),
                    "option_rows": int(option_manifest.get("rows", 0)),
                    "option_source_files": option_manifest.get("files", []),
                    "option_date_from": option_manifest.get("date_from"),
                    "option_date_to": option_manifest.get("date_to"),
                    "option_rows_last_5y": int(option_manifest.get("rows_last_5y", 0) or 0),
                    "option_symbols_last_5y": int(option_manifest.get("symbols_last_5y", 0) or 0),
                    "option_coverage_quality": option_manifest.get("coverage_quality", "none"),
                    "option_coverage_notes": option_manifest.get("coverage_notes", []) or [],
                    "option_dedupe": option_manifest.get("dedupe", {}) or {},
                },
            }
        )
        return payload

    factor_keys = list(default_state()["factor_weights"].keys())
    direction_keys = ["CALL", "PUT"]
    aggregate_factor = {key: 0.0 for key in factor_keys}
    aggregate_direction = {key: 0.0 for key in direction_keys}
    feature_scores: dict[str, list[float]] = {}
    symbol_weights: dict[str, float] = {}
    total_trades = 0
    weighted_return = 0.0
    weighted_win = 0.0
    weighted_confidence = 0.0
    weighted_option_ratio = 0.0
    best_symbol = None
    best_calmar = -9999.0

    for item in valid:
        trade_count = int(item.get("trade_count", 0))
        total_trades += trade_count
        weight = max(1.0, min(6.0, trade_count / 12.0))
        weighted_return += float(item.get("avg_return_pct", 0) or 0) * weight
        weighted_win += float(item.get("win_rate", 0) or 0) * weight
        weighted_confidence += float(item.get("research_confidence", 0) or 0) * weight
        weighted_option_ratio += min(1.0, float(item.get("option_feature_trade_count", 0) or 0) / max(trade_count, 1)) * weight
        symbol = str(item.get("symbol") or "")
        symbol_bias = float(item.get("quality_score", 0) or 0) * 0.8 + float(item.get("calmar", 0) or 0) * 0.35
        symbol_weights[symbol] = round(max(-4.0, min(4.0, symbol_bias)), 2)
        if float(item.get("calmar", -9999) or -9999) > best_calmar:
            best_symbol = symbol
            best_calmar = float(item.get("calmar", -9999) or -9999)
        for key in factor_keys:
            aggregate_factor[key] += float((item.get("factor_weights") or {}).get(key, 0) or 0) * weight
        for key in direction_keys:
            aggregate_direction[key] += float((item.get("direction_scores") or {}).get(key, 0) or 0) * weight
        for key, value in (item.get("feature_weights") or {}).items():
            feature_scores.setdefault(key, []).append(float(value or 0) * weight)

    divisor = max(1.0, sum(max(1.0, min(6.0, int(item.get("trade_count", 0)) / 12.0)) for item in valid))
    factor_weights = {key: round(value / divisor, 2) for key, value in aggregate_factor.items()}
    direction_weights = {key: round(value / divisor, 2) for key, value in aggregate_direction.items()}
    feature_weights = {key: round(sum(values) / divisor, 3) for key, values in feature_scores.items()}

    direction_bias = "balanced"
    if direction_weights["CALL"] - direction_weights["PUT"] >= 0.3:
        direction_bias = "call"
    elif direction_weights["PUT"] - direction_weights["CALL"] >= 0.3:
        direction_bias = "put"

    coverage = {
        "stock_symbols": len(stock_summary.get("symbols", {})),
        "option_symbols": len(option_manifest.get("symbols", {})),
        "stock_rows": int(stock_summary.get("rows", 0)),
        "option_rows": int(option_manifest.get("rows", 0)),
        "option_source_files": option_manifest.get("files", []),
        "option_date_from": option_manifest.get("date_from"),
        "option_date_to": option_manifest.get("date_to"),
        "option_rows_last_5y": int(option_manifest.get("rows_last_5y", 0) or 0),
        "option_symbols_last_5y": int(option_manifest.get("symbols_last_5y", 0) or 0),
        "option_coverage_quality": option_manifest.get("coverage_quality", "none"),
        "option_coverage_notes": option_manifest.get("coverage_notes", []) or [],
        "option_dedupe": option_manifest.get("dedupe", {}) or {},
    }
    mode = _mode_from_option_manifest(option_manifest)
    summary = (
        f"5Y research ready: {len(valid)} symbols, {total_trades} backtest trades, "
        f"direction bias {direction_bias}, best symbol {best_symbol or 'N/A'}, "
        f"option feature coverage {coverage['option_symbols']} symbols, quality {coverage['option_coverage_quality']}."
    )
    return {
        "updated_at": _utc_now_iso(),
        "years": years,
        "horizon_days": horizon_days,
        "threshold": threshold,
        "mode": mode,
        "summary": summary,
        "symbols": symbol_results,
        "aggregate": {
            "symbol_count": len(valid),
            "trade_count": total_trades,
            "avg_return_pct": round(weighted_return / divisor, 2),
            "win_rate": round(weighted_win / divisor, 2),
            "research_confidence": round(weighted_confidence / divisor, 3),
            "option_feature_ratio": round(weighted_option_ratio / divisor, 3),
            "direction_bias": direction_bias,
            "best_symbol": best_symbol,
            "best_calmar": round(best_calmar, 2) if best_symbol else None,
        },
        "factor_weights": factor_weights,
        "direction_weights": direction_weights,
        "symbol_weights": symbol_weights,
        "feature_weights": feature_weights,
        "coverage": coverage,
    }


def run_research(
    base_dir: Path,
    state_path: Path,
    symbols: list[str],
    years: int = 5,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    refresh: bool = False,
    option_source_paths: Optional[list[str]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    skip_stock_backfill: bool = False,
) -> dict[str, Any]:
    symbols = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    if not symbols:
        payload = default_state()
        payload["summary"] = "no symbols provided for historical research"
        return save_state(state_path, payload)

    if not skip_stock_backfill:
        backfill_stock_history(base_dir, symbols, years=years, refresh=refresh)
    stock_summary = backfill_stock_history(base_dir, symbols, years=years, refresh=False)
    option_manifest = import_option_history(base_dir, option_source_paths=option_source_paths, refresh=refresh, symbols=symbols)
    dirs = ensure_dirs(base_dir)

    symbol_results: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        dataset = build_research_dataset(base_dir, symbol, horizon_days=horizon_days)
        if not dataset.empty:
            dataset.reset_index().to_csv(dirs["datasets"] / f"{symbol}.csv", index=False)
        result = backtest_symbol(dataset, symbol, horizon_days=horizon_days, threshold=threshold, min_train_days=MIN_TRAIN_DAYS)
        symbol_results[symbol] = result
        if result.get("status") == "ok":
            equity_curve = result.get("equity_curve", [])
            (dirs["backtests"] / f"{symbol}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if equity_curve:
                pd.DataFrame(equity_curve).to_csv(dirs["backtests"] / f"{symbol}_equity.csv", index=False)

    payload = _aggregate_results(
        symbol_results=symbol_results,
        stock_summary=stock_summary,
        option_manifest=option_manifest,
        years=years,
        horizon_days=horizon_days,
        threshold=threshold,
    )
    return save_state(state_path, payload)


def symbol_snapshot(state: dict[str, Any], symbol: str) -> dict[str, Any]:
    payload = (state.get("symbols") or {}).get(_normalize_symbol(symbol), {}) if isinstance(state, dict) else {}
    return payload if isinstance(payload, dict) else {}


def live_signal_profile(state: dict[str, Any], symbol: str, plan: dict[str, Any], tech_bias: str = "") -> dict[str, Any]:
    if not isinstance(state, dict) or not isinstance(plan, dict):
        return {
            "bonus": 0.0,
            "confidence": 0.0,
            "bias": "n/a",
            "signal": "unavailable",
            "quality_score": 0.0,
            "alignment": "none",
        }
    symbol = _normalize_symbol(symbol)
    symbol_state = symbol_snapshot(state, symbol)
    if not symbol_state or symbol_state.get("status") != "ok":
        aggregate = state.get("aggregate") or {}
        direction = str(plan.get("type") or "").upper()
        factor_weights = state.get("factor_weights") or {}
        raw_bonus = 0.0
        if direction in {"CALL", "PUT"}:
            raw_bonus += float((state.get("direction_weights") or {}).get(direction, 0.0) or 0.0) * 0.22
            direction_bias = str(aggregate.get("direction_bias") or "").lower()
            if direction_bias == "call" and direction == "CALL":
                raw_bonus += 0.18
            elif direction_bias == "put" and direction == "PUT":
                raw_bonus += 0.18
        ivrv_key = f"ivrv_{plan.get('factor_bucket_ivrv')}" if plan.get("factor_bucket_ivrv") else ""
        skew_key = f"skew_{plan.get('factor_bucket_skew')}" if plan.get("factor_bucket_skew") else ""
        flow_key = f"flow_{plan.get('factor_bucket_flow')}" if plan.get("factor_bucket_flow") else ""
        for key, scale in ((ivrv_key, 0.12), (skew_key, 0.14), (flow_key, 0.10)):
            if key and key in factor_weights:
                raw_bonus += float(factor_weights.get(key, 0.0) or 0.0) * scale
        confidence = max(0.0, min(0.45, float(aggregate.get("research_confidence", 0.0) or 0.0) * 0.55))
        bonus = raw_bonus * (0.35 + confidence * 0.65)
        signal = "aggregate_only"
        if bonus >= 0.7:
            signal = "aggregate_support"
        elif bonus <= -0.7:
            signal = "aggregate_contrarian"
        return {
            "bonus": round(max(-2.5, min(2.5, bonus)), 2),
            "confidence": round(confidence, 3),
            "bias": str(aggregate.get("direction_bias") or "balanced"),
            "signal": signal,
            "quality_score": 0.0,
            "alignment": "low" if abs(bonus) > 0.15 else "none",
            "raw_bonus": round(raw_bonus, 2),
        }

    raw_bonus = 0.0
    direction = str(plan.get("type") or "").upper()
    bias = str(symbol_state.get("bias") or "").lower()
    alignment_hits = 0
    if direction in {"CALL", "PUT"}:
        direction_weights = state.get("direction_weights") or {}
        raw_bonus += float(direction_weights.get(direction, 0.0) or 0.0) * 0.55
        if bias == "call" and direction == "CALL":
            raw_bonus += 0.7
            alignment_hits += 1
        elif bias == "put" and direction == "PUT":
            raw_bonus += 0.7
            alignment_hits += 1
        elif bias == "call" and direction == "PUT":
            raw_bonus -= 0.55
        elif bias == "put" and direction == "CALL":
            raw_bonus -= 0.55

    symbol_weight = float((state.get("symbol_weights") or {}).get(symbol, 0.0) or 0.0)
    raw_bonus += symbol_weight * 0.35

    factor_weights = state.get("factor_weights") or {}
    ivrv_key = f"ivrv_{plan.get('factor_bucket_ivrv')}" if plan.get("factor_bucket_ivrv") else ""
    skew_key = f"skew_{plan.get('factor_bucket_skew')}" if plan.get("factor_bucket_skew") else ""
    flow_key = f"flow_{plan.get('factor_bucket_flow')}" if plan.get("factor_bucket_flow") else ""
    for key, scale in ((ivrv_key, 0.32), (skew_key, 0.34), (flow_key, 0.28)):
        if key and key in factor_weights:
            weight = float(factor_weights.get(key, 0.0) or 0.0)
            raw_bonus += weight * scale
            if weight > 0.2:
                alignment_hits += 1

    tech_bias = str(tech_bias or "").lower()
    if direction == "CALL" and "bullish" in tech_bias:
        raw_bonus += 0.2
        alignment_hits += 1
    if direction == "PUT" and "bearish" in tech_bias:
        raw_bonus += 0.2
        alignment_hits += 1

    confidence = float(symbol_state.get("research_confidence", 0.0) or (state.get("aggregate") or {}).get("research_confidence", 0.0) or 0.0)
    quality_score = float(symbol_state.get("quality_score", 0.0) or 0.0)
    scaled_bonus = raw_bonus * (0.45 + confidence * 0.55) + max(-0.8, min(0.8, quality_score * 0.08))
    alignment = "high" if alignment_hits >= 3 else "medium" if alignment_hits >= 2 else "low" if alignment_hits >= 1 else "none"
    if scaled_bonus >= 1.35:
        signal = "supported"
    elif scaled_bonus <= -1.35:
        signal = "contrarian"
    else:
        signal = "mixed"
    return {
        "bonus": round(max(-6.0, min(6.0, scaled_bonus)), 2),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "bias": bias or "balanced",
        "signal": signal,
        "quality_score": round(quality_score, 2),
        "alignment": alignment,
        "raw_bonus": round(raw_bonus, 2),
    }


def forecast_symbol(
    base_dir: Path,
    state: dict[str, Any],
    symbol: str,
    spot: Optional[float] = None,
    horizon_days: int = 5,
    adaptive: bool = False,
    candidate_horizons: Optional[list[int]] = None,
) -> dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    state = _upgrade_state(state if isinstance(state, dict) else {})
    symbol_state = symbol_snapshot(state, symbol)
    requested_horizon = max(3, min(int(horizon_days or 5), 15))
    adaptive_profile = None
    selected_horizon = requested_horizon
    if adaptive:
        adaptive_profile = select_adaptive_horizon(
            base_dir,
            symbol,
            candidate_horizons=candidate_horizons or [3, 5, 7, 10, requested_horizon],
        )
        selected_horizon = int(adaptive_profile.get("selected_horizon_days") or requested_horizon)
    feature_frame = _prepare_feature_frame(base_dir, symbol, horizon_days=selected_horizon, keep_unlabeled=True)
    forecast = _forecast_from_frame(
        state=state,
        symbol_state=symbol_state,
        symbol=symbol,
        feature_frame=feature_frame,
        spot=spot,
        horizon_days=selected_horizon,
    )
    forecast["requested_horizon_days"] = requested_horizon
    forecast["adaptive_horizon"] = bool(adaptive)
    if adaptive_profile:
        forecast["adaptive_profile"] = adaptive_profile
    return forecast


def resolve_forecast_memory(
    base_dir: Path,
    memory_path: Path,
) -> dict[str, Any]:
    payload = load_forecast_memory(memory_path)
    previous_summary = dict(payload.get("summary") or {})
    changed = False
    for item in payload.get("forecasts", []):
        status = str(item.get("status") or "").lower()
        if status not in {"pending", "resolved"}:
            item["status"] = "pending"
            status = "pending"
            changed = True
        if str(item.get("status") or "pending") != "pending":
            continue
        symbol = _normalize_symbol(item.get("symbol"))
        target_date = pd.to_datetime(item.get("target_date"), errors="coerce")
        hist = load_stock_history(base_dir, symbol)
        if pd.isna(target_date) or hist.empty or target_date > hist.index.max():
            continue
        exit_row = hist[hist.index >= target_date]
        if exit_row.empty:
            continue
        actual_close = float(exit_row["Close"].iloc[0])
        entry_spot = float(item.get("spot") or 0.0)
        if entry_spot <= 0 or not np.isfinite(actual_close):
            continue
        actual_move_pct = (actual_close / entry_spot - 1.0) * 100
        predicted_dir = str(item.get("direction") or "neutral").lower()
        success_direction = (
            (predicted_dir == "bullish" and actual_move_pct > 0)
            or (predicted_dir == "bearish" and actual_move_pct < 0)
            or (predicted_dir == "neutral" and abs(actual_move_pct) <= 0.85)
        )
        low = _num(item.get("range_low"), entry_spot)
        high = _num(item.get("range_high"), entry_spot)
        band_hit = min(low, high) <= actual_close <= max(low, high)
        expected_close = _num(item.get("expected_close"), entry_spot)
        item["status"] = "resolved"
        item["resolved_at"] = _utc_now_iso()
        item["actual_close"] = round(actual_close, 2)
        item["actual_move_pct"] = round(actual_move_pct, 2)
        item["success_direction"] = bool(success_direction)
        item["band_hit"] = bool(band_hit)
        item["abs_error_pct"] = round(abs(actual_close - expected_close) / entry_spot * 100, 2)
        changed = True

    forecasts = payload.get("forecasts", [])
    resolved = [item for item in forecasts if str(item.get("status")) == "resolved"]
    payload["summary"] = {
        "total": len(forecasts),
        "resolved": len(resolved),
        "pending": sum(1 for item in forecasts if str(item.get("status") or "pending") == "pending"),
        "direction_accuracy": round(sum(1 for item in resolved if item.get("success_direction")) / len(resolved) * 100, 2) if resolved else None,
        "band_hit_rate": round(sum(1 for item in resolved if item.get("band_hit")) / len(resolved) * 100, 2) if resolved else None,
        "avg_abs_error_pct": round(sum(_num(item.get("abs_error_pct")) for item in resolved) / len(resolved), 2) if resolved else None,
    }
    if changed or payload["summary"] != previous_summary:
        save_forecast_memory(memory_path, payload)
    return payload


def record_forecast(memory_path: Path, forecast: dict[str, Any]) -> dict[str, Any]:
    payload = load_forecast_memory(memory_path)
    forecasts = payload.get("forecasts", [])
    key = (
        _normalize_symbol(forecast.get("symbol")),
        str(forecast.get("forecast_date") or ""),
        int(forecast.get("horizon_days") or 0),
    )
    exists = {
        (
            _normalize_symbol(item.get("symbol")),
            str(item.get("forecast_date") or ""),
            int(item.get("horizon_days") or 0),
        )
        for item in forecasts
    }
    if key not in exists:
        item = dict(forecast)
        item["id"] = item.get("id") or uuid.uuid4().hex
        item["status"] = "pending"
        item["created_at"] = item.get("created_at") or _utc_now_iso()
        forecasts.append(item)
    payload["forecasts"] = forecasts[-800:]
    save_forecast_memory(memory_path, payload)
    return resolve_forecast_memory(memory_path.parent, memory_path)


def compute_live_bonus(state: dict[str, Any], symbol: str, plan: dict[str, Any], tech_bias: str = "") -> float:
    return float(live_signal_profile(state, symbol, plan, tech_bias).get("bonus", 0.0) or 0.0)
