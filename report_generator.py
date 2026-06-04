from __future__ import annotations

import datetime
import logging
import os
import re
from typing import Any

import pandas as pd
from tabulate import tabulate

from config import REPORT_DIR

logger = logging.getLogger(__name__)


def _safe_session_label(session_label: str) -> str:
    label = str(session_label or "").strip()
    folded = label.lower().replace(" ", "_").replace("-", "_")
    if any(token in folded for token in ("premarket", "pre_market", "盘前", "鐩樺墠")):
        return "premarket"
    if any(token in folded for token in ("postmarket", "post_market", "盘后", "鐩樺悗")):
        return "postmarket"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label.replace(" ", "_"))
    slug = slug.strip("._-")
    return slug or "session"


def _ensure_dir() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)


def df_to_table(df: pd.DataFrame, max_rows: int = 5) -> str:
    if df is None or df.empty:
        return "  (暂无数据)"
    return tabulate(df.head(max_rows), headers="keys", tablefmt="simple", showindex=False)


def _fmt(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _fmt_money(value: Any) -> str:
    try:
        amount = float(value or 0)
    except Exception:
        return "N/A"
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:.0f}"


def build_report(
    symbol: str,
    stock_info: dict,
    iv_summary: dict,
    greeks_summary: dict,
    top_iv_df: pd.DataFrame,
    top_oi_df: pd.DataFrame,
    timestamp: str,
) -> str:
    lines = [
        "=" * 50,
        f"  期权分析报告 · {symbol}",
        f"  {timestamp}",
        "=" * 50,
        "",
        "【行情概览】",
        f"  现价：${_fmt(stock_info.get('price'))}",
        f"  52周高/低：${_fmt(stock_info.get('week52_high'))} / ${_fmt(stock_info.get('week52_low'))}",
        f"  HV30（历史波动率）：{_fmt(stock_info.get('hv30'))}%",
        "",
        "【IV 分析】",
        f"  ATM IV：{_fmt(iv_summary.get('atm_iv'))}%",
        f"  IV 均值：{_fmt(iv_summary.get('iv_mean'))}%",
        f"  IV Rank：{_fmt(iv_summary.get('iv_rank'))}（0=历史低位 / 100=历史高位）",
        f"  IV Skew（Put-Call）：{_fmt(iv_summary.get('iv_skew'))}%",
        "",
        "【策略信号】",
        f"  {_fmt(iv_summary.get('signal'))}",
        "",
        "【Greeks 均值】",
        f"  Delta (Call/Put)：{_fmt(greeks_summary.get('avg_delta_call'))} / {_fmt(greeks_summary.get('avg_delta_put'))}",
        f"  Theta（每日）：{_fmt(greeks_summary.get('avg_theta'))}",
        f"  Vega（每1%IV）：{_fmt(greeks_summary.get('avg_vega'))}",
        "",
        "【高IV合约 TOP5】",
        df_to_table(top_iv_df),
        "",
        "【高OI主力合约 TOP5】",
        df_to_table(top_oi_df),
        "",
        "=" * 50,
        "  Powered by 智能期权分析器",
    ]
    return "\n".join(lines)


def save_report(symbol: str, report_text: str, chain_df: pd.DataFrame) -> None:
    _ensure_dir()
    date_str = datetime.date.today().isoformat()
    txt_path = os.path.join(REPORT_DIR, f"{symbol}_{date_str}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("报告已保存：%s", txt_path)

    if chain_df is not None and not chain_df.empty:
        csv_path = os.path.join(REPORT_DIR, f"{symbol}_{date_str}_chain.csv")
        chain_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("期权链已保存：%s", csv_path)


def save_daily_report(report_text: str, session_label: str, date_str: str | None = None) -> None:
    _ensure_dir()
    date_str = date_str or datetime.date.today().isoformat()
    safe_label = _safe_session_label(session_label)
    txt_path = os.path.join(REPORT_DIR, f"daily_{safe_label}_{date_str}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("会话报告已保存：%s", txt_path)


def build_premarket_report(
    recommendations: list[dict],
    timestamp: str,
    session_label: str,
    user_bias: dict | None = None,
    market_context: dict | None = None,
    unusual_rows: list[dict] | None = None,
    focus_profile: dict | None = None,
    distillate: dict | None = None,
) -> str:
    user_bias = user_bias or {}
    market_context = market_context or {}
    unusual_rows = unusual_rows or []
    focus_profile = focus_profile or {}
    distillate = distillate or {}

    focus_symbols = [str(symbol) for symbol in (focus_profile.get("focus_symbols") or []) if str(symbol)]
    focus_summary = list(focus_profile.get("summary") or [])
    focus_cluster_lines = [
        f"{item.get('cluster')} x{item.get('count')}"
        for item in (focus_profile.get("top_clusters") or [])[:3]
        if item.get("cluster")
    ]
    distillate_essence = [str(item) for item in (distillate.get("essence") or []) if str(item)]
    distillate_playbook = [str(item) for item in (distillate.get("playbook") or []) if str(item)]
    distillate_avoid = [str(item) for item in (distillate.get("avoid") or []) if str(item)]
    distillate_focus = [
        str(item.get("symbol"))
        for item in (distillate.get("focus_symbols") or [])
        if isinstance(item, dict) and item.get("symbol")
    ]
    distillate_pending = [
        f"{str(item.get('symbol') or '').upper()} {str(item.get('type') or '').upper()}".strip()
        for item in (distillate.get("pending_signals") or [])[:4]
        if isinstance(item, dict) and item.get("symbol")
    ]
    market_heat = market_context.get("market_heat") or distillate.get("market_heat") or {}
    market_heat_digest = [
        str(item.get("summary") or "").strip()
        for item in (market_heat.get("top_sectors") or [])[:3]
        if isinstance(item, dict) and str(item.get("summary") or "").strip()
    ]

    lines = [
        "## 盘前报告",
        f"**阶段**：{session_label}",
        f"**时间**：{timestamp}",
        "",
        "### 偏好画像",
        (
            f"> 方向 { _fmt(user_bias.get('direction', 'neutral')) } | "
            f"DTE { _fmt(user_bias.get('dte_style', 'mixed')) } | "
            f"IV { _fmt(user_bias.get('iv_style', 'balanced')) } | "
            f"结构 { _fmt(user_bias.get('structure', 'both')) }"
        ),
        "",
        "### 市场热点",
        f"> 板块：{_fmt(market_context.get('sector_line'), '暂无')}",
        f"> 新闻：{_fmt(market_context.get('news_line'), '暂无')}",
        f"> 资金：{_fmt(market_context.get('flow_line'), '暂无')}",
    ]

    if focus_symbols or focus_summary or focus_cluster_lines:
        lines.extend(["", "### 关注池"])
        for item in focus_summary[:3]:
            lines.append(f"> {item}")
        if focus_symbols:
            lines.append(f"> 标的：{', '.join(focus_symbols[:8])}")
        if focus_cluster_lines:
            lines.append(f"> 聚类：{' / '.join(focus_cluster_lines)}")

    if distillate_essence or distillate_playbook or distillate_avoid or distillate_focus or distillate_pending:
        lines.extend(["", "### 策略蒸馏"])
        for item in distillate_essence[:3]:
            lines.append(f"> 结论：{item}")
        for item in distillate_playbook[:3]:
            lines.append(f"> 执行：{item}")
        for item in distillate_avoid[:2]:
            lines.append(f"> 规避：{item}")
        if distillate_focus:
            lines.append(f"> 焦点：{', '.join(distillate_focus[:8])}")
        if distillate_pending:
            lines.append(f"> 待验证：{', '.join(distillate_pending[:4])}")
    if market_heat_digest:
        lines.extend(["", "### 热点摘要"])
        for item in market_heat_digest:
            lines.append(f"> {item}")

    lines.extend(["", "### 候选建议"])

    if not recommendations:
        lines.append("> 暂无候选")
    for idx, rec in enumerate(recommendations, start=1):
        focus_marker = ""
        if str(rec.get("symbol") or "") in focus_symbols:
            focus_marker = " [关注]"
        lines.extend(
            [
                f"{idx}. **{_fmt(rec.get('symbol'), '?')}{focus_marker}** `{_fmt(rec.get('contract'), '')}` | {_fmt(rec.get('type'))} | Score {_fmt(rec.get('score'))}",
                f"   - 入场 {_fmt(rec.get('entry'))} / 止损 {_fmt(rec.get('stop_loss'))} / 止盈 {_fmt(rec.get('take_profit'))}",
                f"   - 触发 {_fmt(rec.get('trigger'))} / 失效 {_fmt(rec.get('invalidation'))}",
                f"   - 板块 {_fmt(rec.get('sector'), 'N/A')} | 理由 {_fmt(rec.get('reason'), 'N/A')}",
                f"   - 新闻 {_fmt(rec.get('news_summary'), '暂无新闻摘要')}",
            ]
        )

    if unusual_rows:
        lines.extend(["", "### 异常期权"])
        for idx, row in enumerate(unusual_rows[:8], start=1):
            plan = row.get("best_plan", {}) if isinstance(row.get("best_plan"), dict) else {}
            contract = plan.get("contract") or row.get("contract", {}).get("name") or ""
            lines.extend(
                [
                    f"{idx}. **{_fmt(row.get('symbol'), '?')} {contract}** | 分数 {_fmt(row.get('final_score'))}",
                    f"   - Vol/OI {_fmt(row.get('vol_oi_ratio'))}x | 成交额 {_fmt_money(row.get('premium'))} | IV {_fmt(row.get('iv_pct'))}%",
                    f"   - 入场 {_fmt(plan.get('entry'))} / 止损 {_fmt(plan.get('stop_loss'))} / 止盈 {_fmt(plan.get('take_profit'))}",
                ]
            )

    lines.extend(["", "_以上内容用于盘前准备，不构成实盘执行保证。_"])
    return "\n".join(lines)


def build_postmarket_review_report(
    timestamp: str,
    session_label: str,
    learning_profile: dict | None = None,
    sim_summary: dict | None = None,
    resolved_signals: list[dict] | None = None,
    next_watchlist: list[dict] | None = None,
    focus_profile: dict | None = None,
    distillate: dict | None = None,
) -> str:
    learning_profile = learning_profile or {}
    sim_summary = sim_summary or {}
    resolved_signals = resolved_signals or []
    next_watchlist = next_watchlist or []
    focus_profile = focus_profile or {}
    distillate = distillate or {}

    focus_symbols = [str(symbol) for symbol in (focus_profile.get("focus_symbols") or []) if str(symbol)]
    focus_summary = list(focus_profile.get("summary") or [])
    distillate_essence = [str(item) for item in (distillate.get("essence") or []) if str(item)]
    distillate_playbook = [str(item) for item in (distillate.get("playbook") or []) if str(item)]
    distillate_avoid = [str(item) for item in (distillate.get("avoid") or []) if str(item)]
    distillate_pending_forecasts = [
        f"{str(item.get('symbol') or '').upper()} {item.get('direction') or ''}".strip()
        for item in (distillate.get("pending_forecasts") or [])[:4]
        if isinstance(item, dict) and item.get("symbol")
    ]
    market_heat = distillate.get("market_heat") or {}
    market_heat_digest = [
        str(item.get("summary") or "").strip()
        for item in (market_heat.get("top_sectors") or [])[:3]
        if isinstance(item, dict) and str(item.get("summary") or "").strip()
    ]

    lines = [
        "## 盘后复盘",
        f"**阶段**：{session_label}",
        f"**时间**：{timestamp}",
        "",
        "### 学习概览",
        f"> 摘要：{_fmt(learning_profile.get('summary'), '暂无')}",
        f"> 已平仓 { _fmt(learning_profile.get('closed_count'), 0) } | 已验证信号 { _fmt(learning_profile.get('resolved_signal_count'), 0) } | 置信度 { _fmt(learning_profile.get('confidence'), 0) }",
        "",
        "### 模拟仓 / 收益",
        (
            f"> Open {_fmt(sim_summary.get('open_count'), 0)} | Closed {_fmt(sim_summary.get('closed_count'), 0)} | "
            f"Realized {_fmt(sim_summary.get('realized_pnl'), 0)} | Unrealized {_fmt(sim_summary.get('open_unrealized_pnl'), 0)} | "
            f"WinRate {_fmt(sim_summary.get('win_rate'), 0)}%"
        ),
    ]

    if focus_symbols or focus_summary:
        lines.extend(["", "### 关注池"])
        for item in focus_summary[:2]:
            lines.append(f"> {item}")
        if focus_symbols:
            lines.append(f"> 标的：{', '.join(focus_symbols[:8])}")

    if distillate_essence or distillate_playbook or distillate_avoid or distillate_pending_forecasts:
        lines.extend(["", "### 策略蒸馏"])
        for item in distillate_essence[:3]:
            lines.append(f"> 结论：{item}")
        for item in distillate_playbook[:3]:
            lines.append(f"> 执行：{item}")
        for item in distillate_avoid[:2]:
            lines.append(f"> 规避：{item}")
        if distillate_pending_forecasts:
            lines.append(f"> 待验证：{', '.join(distillate_pending_forecasts[:4])}")
    if market_heat_digest:
        lines.extend(["", "### 热点摘要"])
        for item in market_heat_digest:
            lines.append(f"> {item}")

    if resolved_signals:
        lines.extend(["", "### 已验证信号"])
        for idx, item in enumerate(resolved_signals[:10], start=1):
            outcome = "成功" if item.get("success") else "失败"
            lines.extend(
                [
                    f"{idx}. **{_fmt(item.get('symbol'), '?')} {_fmt(item.get('type'))} {outcome}**",
                    f"   - 正股收益 {_fmt(item.get('underlying_return_pct'))}% | 方向优势 {_fmt(item.get('directional_edge_pct'))}%",
                    f"   - 因子 IV-HV {_fmt(item.get('factor_bucket_ivrv'))} | Skew {_fmt(item.get('factor_bucket_skew'))} | Flow {_fmt(item.get('factor_bucket_flow'))} | Liquidity {_fmt(item.get('factor_bucket_liquidity'))}",
                ]
            )
    else:
        lines.extend(["", "### 已验证信号", "> 暂无可复盘的已验证信号"])

    if next_watchlist:
        lines.extend(["", "### 下轮观察"])
        for idx, rec in enumerate(next_watchlist[:5], start=1):
            focus_marker = " [关注]" if str(rec.get("symbol") or "") in focus_symbols else ""
            lines.extend(
                [
                    f"{idx}. **{_fmt(rec.get('symbol'), '?')}{focus_marker}** `{_fmt(rec.get('contract'), '')}` | {_fmt(rec.get('type'))} | Score {_fmt(rec.get('score'))}",
                    f"   - 触发 {_fmt(rec.get('trigger'))} / 失效 {_fmt(rec.get('invalidation'))}",
                    f"   - 理由 {_fmt(rec.get('reason'), 'N/A')}",
                ]
            )

    lines.extend(["", "_以上内容用于盘后复盘和下一轮观察，不构成实盘执行保证。_"])
    return "\n".join(lines)


def build_summary_text(all_reports: list[dict]) -> str:
    if not all_reports:
        return "本次期权分析无有效数据。"
    lines = ["## 美股期权每日分析汇总", ""]
    for r in all_reports:
        lines.append(f"**{_fmt(r.get('symbol'), '?')}** 现价 ${_fmt(r.get('price'))} | ATM IV {_fmt(r.get('atm_iv'))}% | IVR {_fmt(r.get('iv_rank'))}")
        lines.append(f"> {_fmt(r.get('signal'), '暂无信号')}")
    lines.append("")
    lines.append("_仅供参考，不构成投资建议。_")
    return "\n".join(lines)
