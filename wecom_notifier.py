from __future__ import annotations

import logging
import time

import requests

from config import WECOM_WEBHOOK_URL

logger = logging.getLogger(__name__)

MAX_CHARS = 4096
MAX_MARKDOWN_CHARS = 3500
REQUEST_TIMEOUT = 12
RETRY_DELAYS = (1.0, 2.5)


def send_markdown(content: str) -> bool:
    if not _configured():
        logger.warning("企业微信 Webhook 未配置，跳过推送")
        return False
    chunks = _split_content(content, MAX_MARKDOWN_CHARS)
    success = True
    for idx, chunk in enumerate(chunks, start=1):
        ok = _post_markdown(chunk)
        if not ok:
            logger.error("企业微信 markdown 分片发送失败 %s/%s", idx, len(chunks))
            success = False
    return success


def send_text(content: str) -> bool:
    if not _configured():
        logger.warning("企业微信 Webhook 未配置，跳过推送")
        return False
    chunks = _split_content(content, MAX_CHARS)
    success = True
    for idx, chunk in enumerate(chunks, start=1):
        ok = _post_text(chunk)
        if not ok:
            logger.error("企业微信 text 分片发送失败 %s/%s", idx, len(chunks))
            success = False
    return success


def _configured() -> bool:
    return bool(WECOM_WEBHOOK_URL and "YOUR_KEY_HERE" not in WECOM_WEBHOOK_URL)


def _post_markdown(content: str) -> bool:
    return _post({"msgtype": "markdown", "markdown": {"content": content}})


def _post_text(content: str) -> bool:
    return _post({"msgtype": "text", "text": {"content": content}})


def _post(payload: dict) -> bool:
    last_error = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            resp = requests.post(WECOM_WEBHOOK_URL, json=payload, timeout=REQUEST_TIMEOUT)
            data = resp.json()
            if data.get("errcode") == 0:
                logger.info("企业微信消息发送成功")
                return True
            last_error = f"企业微信返回错误: {data}"
            logger.error(last_error)
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return False
        except Exception as exc:
            last_error = str(exc)
            logger.error("企业微信推送异常: %s", exc)
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return False
    logger.error("企业微信推送失败: %s", last_error)
    return False


def _split_content(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len and len(text.encode("utf-8")) <= max_len:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        line_len = len(line)
        line_bytes = len(line.encode("utf-8"))
        current_bytes = len("".join(current).encode("utf-8")) if current else 0
        if current and (current_len + line_len > max_len or current_bytes + line_bytes > max_len):
            chunks.append("".join(current))
            current = []
            current_len = 0
        if line_len > max_len or line_bytes > max_len:
            for piece in _split_long_line(line, max_len):
                if current:
                    chunks.append("".join(current))
                    current = []
                    current_len = 0
                chunks.append(piece)
            continue
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("".join(current))
    return chunks


def _split_long_line(text: str, max_len: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if current and (len(candidate) > max_len or len(candidate.encode("utf-8")) > max_len):
            pieces.append(current)
            current = ch
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces
