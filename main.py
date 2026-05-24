from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pytz
import schedule
import yfinance as yf

import app as webapp
from config import DAILY_PRESELECT_LIMIT, DAILY_REPORT_LIMIT, SCHEDULE_TIMES_ET, SYMBOLS
from data_fetcher import get_company_profile, get_option_chain, get_price_history, get_recent_news, get_stock_info
from greeks_calculator import calculate_greeks_df, summarize_greeks
from iv_analyzer import analyze_iv, high_oi_contracts, top_iv_contracts
from report_generator import (
    build_postmarket_review_report,
    build_premarket_report,
    build_report,
    build_summary_text,
    save_daily_report,
    save_report,
)
from wecom_notifier import send_markdown, send_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("options_analyzer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
SESSION_CACHE_DIR = Path("cache/session_reports")
SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

POS_WORDS = {
    "beat", "beats", "raised", "upgrade", "upgraded", "growth", "record", "surge",
    "rally", "strong", "guidance", "approval", "partnership", "ai", "buy", "outperform",
}
NEG_WORDS = {
    "miss", "misses", "cut", "downgrade", "downgraded", "lawsuit", "probe", "weak",
    "slump", "recall", "delay", "drop", "selloff", "underperform", "lower",
}


def _session_kind(session_label: str) -> str:
    label = str(session_label or "").strip()
    folded = label.lower().replace(" ", "_").replace("-", "_")
    if any(token in folded for token in ("premarket", "pre_market", "盘前", "鐩樺墠")):
        return "premarket"
    if any(token in folded for token in ("postmarket", "post_market", "盘后", "鐩樺悗")):
        return "postmarket"
    return "session"


def _safe_console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode("gbk", errors="replace").decode("gbk", errors="replace")
        print(safe_text)


def _is_us_market_day() -> bool:
    return datetime.datetime.now(ET).weekday() < 5


def _now_et() -> datetime.datetime:
    return datetime.datetime.now(ET)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _session_cache_path(session_label: str) -> Path:
    date_key = _now_et().date().isoformat()
    safe_label = _session_cache_key(session_label)
    return SESSION_CACHE_DIR / f"{date_key}_{safe_label}.json"


def _session_cache_key(session_label: str) -> str:
    label = str(session_label or "").strip()
    kind = _session_kind(label)
    if kind != "session":
        return kind
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label.replace(" ", "_"))
    slug = slug.strip("._-")
    return slug or "session"


def _load_session_cache(session_label: str, max_age_minutes: float) -> dict | None:
    path = _session_cache_path(session_label)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        created_at = datetime.datetime.fromisoformat(str(payload.get("created_at")))
        now = _now_et()
        if created_at.tzinfo is None:
            created_at = ET.localize(created_at)
        else:
            created_at = created_at.astimezone(ET)
        age_minutes = (now - created_at).total_seconds() / 60.0
        if age_minutes > max_age_minutes:
            return None
        return payload
    except Exception as exc:
        logger.warning("session cache load failed: %s", exc)
        return None


def _save_session_cache(session_label: str, payload: dict) -> None:
    _prune_session_cache(max_days=5)
    path = _session_cache_path(session_label)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prune_session_cache(max_days: int = 5) -> None:
    cutoff = _now_et() - datetime.timedelta(days=max_days)
    for path in SESSION_CACHE_DIR.glob("*.json"):
        try:
            modified = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=ET)
            if modified < cutoff:
                path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("session cache prune skipped for %s: %s", path, exc)


def _news_sentiment(news_items: list[dict]) -> tuple[float, str]:
    if not news_items:
        return 0.0, "无新闻"
    score = 0.0
    headlines: list[str] = []
    for item in news_items[:4]:
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        item_score = 0.0
        for word in POS_WORDS:
            if word in text:
                item_score += 1.0
        for word in NEG_WORDS:
            if word in text:
                item_score -= 1.0
        score += item_score
        title = item.get("title") or item.get("summary") or ""
        if title:
            headlines.append(title[:70])
    return _clamp(score, -5.0, 5.0), "；".join(headlines[:2]) if headlines else "无标题"


def _sector_score(sector: str) -> tuple[float, str]:
    etf = SECTOR_ETF_MAP.get(sector or "")
    if not etf:
        return 0.0, sector or "Unknown"
    try:
        hist, _source = webapp._daily_history(etf, 30)
    except Exception as exc:
        logger.debug("%s sector etf failed: %s", etf, exc)
        return 0.0, f"{sector}({etf}) 数据不足"
    if hist.empty or len(hist) < 6:
        return 0.0, f"{sector}({etf}) 数据不足"
    close = hist["Close"].dropna()
    if len(close) < 6:
        return 0.0, f"{sector}({etf}) 数据不足"
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else ret5
    score = _clamp(ret5 * 0.6 + ret20 * 0.3, -6.0, 6.0)
    return score, f"{sector}({etf}) 5日 {ret5:+.1f}% / 20日 {ret20:+.1f}%"


def _fund_flow_score(symbol: str) -> tuple[float, str]:
    try:
        hist, _source = webapp._daily_history(symbol, 30)
    except Exception as exc:
        logger.debug("%s flow failed: %s", symbol, exc)
        return 0.0, "成交量数据不足"
    if hist is None or hist.empty or "Volume" not in hist.columns:
        return 0.0, "成交量数据不足"
    vol = hist["Volume"].dropna()
    close = hist["Close"].dropna()
    if len(vol) < 5 or len(close) < 5:
        return 0.0, "成交量数据不足"
    avg20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    last_vol = float(vol.iloc[-1])
    rel_vol = last_vol / avg20 if avg20 > 0 else 1.0
    ret5 = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0.0
    score = _clamp((rel_vol - 1.0) * 4.0 + ret5 * 0.2, -5.0, 5.0)
    return score, f"相对成交量 {rel_vol:.2f}x，5日涨跌 {ret5:+.1f}%"


def _format_reason(row: dict, news_summary: str, sector_summary: str, flow_summary: str) -> str:
    plan = row.get("best_plan", {}) or {}
    parts = [
        f"基础评分 {float(row.get('final_score', 0) or 0):.2f}",
        f"风格匹配 {float(row.get('style_bonus', 0) or 0):+.2f}",
        f"RR {plan.get('risk_reward', 'N/A')}",
        f"新闻 {news_summary}",
        f"板块 {sector_summary}",
        f"资金 {flow_summary}",
    ]
    return "；".join(parts)


def _build_market_context(rows: list[dict]) -> dict:
    sector_counter = Counter()
    news_pieces: list[str] = []
    flow_pieces: list[str] = []
    for row in rows[:DAILY_REPORT_LIMIT]:
        sector = row.get("sector") or "Unknown"
        if sector != "Unknown":
            sector_counter[sector] += 1
        if row.get("news_summary"):
            news_pieces.append(row["news_summary"])
        if row.get("flow_summary"):
            flow_pieces.append(row["flow_summary"])
    return {
        "sector_line": " / ".join([f"{k} x{v}" for k, v in sector_counter.most_common(3)]) if sector_counter else "无明显板块集中",
        "news_line": "；".join(news_pieces[:2]) if news_pieces else "无明显新闻驱动",
        "flow_line": "；".join(flow_pieces[:2]) if flow_pieces else "无明显资金集中",
    }


def _build_watchlist_snapshot() -> tuple[list[dict], dict, dict]:
    universe = webapp._analyze_marketcap_universe(
        limit=DAILY_PRESELECT_LIMIT,
        universe_size=240,
        refresh=False,
        use_library_bias=True,
        cache_max_age_hours=6,
    )
    if "error" in universe:
        raise RuntimeError(universe["error"])

    conclusion = universe.get("conclusion", {}) or {}
    bias = {
        "direction": conclusion.get("direction", "neutral"),
        "dte_style": conclusion.get("dte_style", "mixed"),
        "iv_style": conclusion.get("iv_style", "balanced"),
        "structure": conclusion.get("structure", "both"),
    }

    ranked: list[dict] = []
    candidate_rows = universe.get("rows", [])[: max(DAILY_REPORT_LIMIT, 8)]
    for row in candidate_rows:
        symbol = row.get("symbol")
        if not symbol:
            continue
        try:
            base_score = float(row.get("final_score", 0) or 0)
            sector = row.get("sector") or "Unknown"
            news_summary = row.get("news_headline") or ""
            sector_summary = sector
            flow_summary = f"技术偏向 {row.get('tech_bias') or 'neutral'} / IVR {row.get('iv_rank') or 'N/A'}"
            final_score = round(base_score, 2)
            plan = row.get("best_plan", {}) or {}
            ranked.append(
                {
                    "symbol": symbol,
                    "name": row.get("name") or row.get("company_name") or symbol,
                    "sector": sector,
                    "contract": plan.get("contract", ""),
                    "type": plan.get("type", ""),
                    "entry": plan.get("entry"),
                    "stop_loss": plan.get("stop_loss"),
                    "take_profit": plan.get("take_profit"),
                    "trigger": plan.get("underlying_trigger"),
                    "invalidation": plan.get("underlying_invalidation"),
                    "score": final_score,
                    "base_score": base_score,
                    "style_bonus": row.get("style_bonus", 0),
                    "reason": row.get("reason_cn") or _format_reason(row, news_summary, sector_summary, flow_summary),
                    "news_summary": news_summary,
                    "sector_summary": sector_summary,
                    "flow_summary": flow_summary,
                    "best_plan": plan,
                }
            )
        except Exception as exc:
            logger.warning("%s daily ranking failed: %s", symbol, exc)

    ranked = sorted(ranked, key=lambda item: item["score"], reverse=True)[:DAILY_REPORT_LIMIT]
    return ranked, bias, _build_market_context(ranked)


def _load_unusual_watchlist() -> dict:
    try:
        cached = webapp._load_latest_nonempty_unusual_cache(max_age_hours=24)
        if cached:
            return cached
        return webapp._scan_daily_unusual_options(
            limit=10,
            universe_size=240,
            refresh=False,
            min_dte=1,
            max_dte=45,
            min_volume=50,
            min_premium=25000,
            max_symbols=18,
            top_per_symbol=4,
            use_library_bias=True,
            cache_max_age_hours=6,
            prefilter_multiplier=2.5,
        )
    except Exception as exc:
        logger.warning("daily unusual options scan failed: %s", exc)
        return {"rows": [], "returned": 0}


def build_premarket_prediction_report(session_label: str, use_cache: bool = True) -> tuple[str, list[dict], dict, dict]:
    if use_cache:
        cached = _load_session_cache(session_label, max_age_minutes=45)
        if cached and cached.get("report_text"):
            return cached["report_text"], cached.get("ranked", []), cached.get("bias", {}), cached.get("market_context", {})

    ranked, bias, market_context = _build_watchlist_snapshot()
    unusual = _load_unusual_watchlist()
    market_context["unusual_count"] = unusual.get("returned", 0)
    report_text = build_premarket_report(
        recommendations=ranked,
        timestamp=_now_et().strftime("%Y-%m-%d %H:%M ET"),
        session_label=session_label,
        user_bias=bias,
        market_context=market_context,
        unusual_rows=unusual.get("rows") or [],
    )
    _save_session_cache(
        session_label,
        {
            "created_at": _now_et().isoformat(),
            "report_text": report_text,
            "ranked": ranked,
            "bias": bias,
            "market_context": market_context,
        },
    )
    return report_text, ranked, bias, market_context


def _recent_resolved_signals(limit: int = 10) -> list[dict]:
    signal_state = webapp._refresh_signal_memory()
    resolved = [item for item in signal_state.get("signals", []) if item.get("status") == "resolved"]
    resolved.sort(key=lambda item: str(item.get("resolved_at") or item.get("scan_date") or ""), reverse=True)
    return resolved[:limit]


def build_postmarket_review(session_label: str, use_cache: bool = True) -> tuple[str, dict]:
    if use_cache:
        cached = _load_session_cache(session_label, max_age_minutes=90)
        if cached and cached.get("report_text"):
            return cached["report_text"], cached.get("payload", {})

    ranked, _bias, _market_context = _build_watchlist_snapshot()
    learning_profile = webapp._adaptive_learning_profile()
    sim_state = webapp._refresh_sim_state(webapp._load_sim_state())
    sim_summary = webapp._summarize_sim_state(sim_state)
    resolved_signals = _recent_resolved_signals(limit=10)
    report_text = build_postmarket_review_report(
        timestamp=_now_et().strftime("%Y-%m-%d %H:%M ET"),
        session_label=session_label,
        learning_profile=learning_profile,
        sim_summary=sim_summary,
        resolved_signals=resolved_signals,
        next_watchlist=ranked[:5],
    )
    payload = {
        "learning_profile": learning_profile,
        "sim_summary": sim_summary,
        "resolved_signals": resolved_signals,
        "next_watchlist": ranked[:5],
    }
    _save_session_cache(
        session_label,
        {
            "created_at": _now_et().isoformat(),
            "report_text": report_text,
            "payload": payload,
        },
    )
    return report_text, payload


def send_session_report(session_label: str, use_cache: bool = True) -> bool:
    if _session_kind(session_label) == "premarket":
        report_text, ranked, _bias, _market_context = build_premarket_prediction_report(session_label, use_cache=use_cache)
        logger.info("盘前预测报告完成，候选数=%s，session=%s", len(ranked), session_label)
    else:
        report_text, payload = build_postmarket_review(session_label, use_cache=use_cache)
        logger.info("盘后复盘报告完成，resolved=%s，session=%s", len(payload.get("resolved_signals", [])), session_label)

    save_daily_report(report_text, session_label)
    try:
        webapp._append_session_report_to_knowledge_wiki(session_label, report_text, timestamp=webapp._now_et_iso())
    except Exception as exc:
        logger.warning("knowledge wiki session append failed: %s", exc)
    ok = send_markdown(report_text)
    if not ok:
        logger.warning("企业微信 markdown 推送失败，回退 text")
        ok = send_text(report_text)
    return ok


def analyze_symbol(symbol: str) -> dict | None:
    logger.info("开始分析 %s ...", symbol)
    try:
        stock_info = get_stock_info(symbol)
        logger.info("  %s 现价 $%s，HV30 %s%%", symbol, stock_info["price"], stock_info["hv30"])
        chain_df = get_option_chain(symbol)
        if chain_df is None or chain_df.empty:
            logger.warning("  %s 期权链为空，跳过", symbol)
            return None
        chain_df = calculate_greeks_df(chain_df)
        iv_summary = analyze_iv(chain_df)
        greeks_summary = summarize_greeks(chain_df)
        top_iv_df = top_iv_contracts(chain_df, n=5)
        top_oi_df = high_oi_contracts(chain_df, n=5)
        report_text = build_report(symbol, stock_info, iv_summary, greeks_summary, top_iv_df, top_oi_df, _now_et().strftime("%Y-%m-%d %H:%M ET"))
        save_report(symbol, report_text, chain_df)
        _safe_console_print("\n" + report_text + "\n")
        return {
            "symbol": symbol,
            "price": stock_info["price"],
            "atm_iv": iv_summary.get("atm_iv"),
            "iv_rank": iv_summary.get("iv_rank"),
            "signal": iv_summary.get("signal", ""),
        }
    except Exception as exc:
        logger.error("  %s 分析失败: %s", symbol, exc, exc_info=True)
        return None


def run_analysis() -> None:
    logger.info("=" * 60)
    logger.info("开始全量期权分析 %s", _now_et().strftime("%Y-%m-%d %H:%M ET"))
    logger.info("=" * 60)
    all_results = []
    for sym in SYMBOLS:
        result = analyze_symbol(sym)
        if result:
            all_results.append(result)
    if all_results:
        send_markdown(build_summary_text(all_results))
    else:
        send_text("本次期权分析无有效数据，请检查网络或 API")
    logger.info("全量分析完成")


def _scheduled_job(session_label: str) -> None:
    if _is_us_market_day():
        send_session_report(session_label, use_cache=True)
    else:
        logger.info("今日非交易日，跳过报告发送")


def start_scheduler() -> None:
    for t in SCHEDULE_TIMES_ET:
        session_label = "盘前预测" if t == "08:30" else "盘后复盘"
        schedule.every().day.at(t).do(_scheduled_job, session_label=session_label)
        logger.info("已注册定时任务：每天美东时间 %s (%s)", t, session_label)
    logger.info("定时调度器已启动，监控标的：%s", ", ".join(SYMBOLS))
    logger.info("按 Ctrl+C 退出\n")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="每日期权报告调度器")
    parser.add_argument("--now", action="store_true", help="立即运行一次全量分析")
    parser.add_argument("--daily", action="store_true", help="立即发送一次会话报告")
    parser.add_argument("--symbol", type=str, default=None, help="只分析单一标的，例如 --symbol AAPL")
    parser.add_argument("--session", type=str, default="手动", help="报告时段标签，如 盘前预测/盘后复盘")
    parser.add_argument("--refresh", action="store_true", help="忽略会话缓存，强制重算报告")
    args = parser.parse_args()

    if args.symbol:
        result = analyze_symbol(args.symbol.upper())
        if result:
            send_markdown(build_summary_text([result]))
    elif args.daily:
        send_session_report(args.session, use_cache=not args.refresh)
    elif args.now:
        run_analysis()
    else:
        start_scheduler()
