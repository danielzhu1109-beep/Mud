# ============================================================
#  config.py  -  智能期权分析器配置
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- 分析标的 ----------
SYMBOLS = ["SPY", "AAPL", "TSLA", "QQQ", "NVDA"]

# ---------- 企业微信 Webhook ----------
# 去「企业微信 -> 群机器人 -> 添加机器人」复制 Webhook URL
# 可放进 .env 文件：WECOM_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
WECOM_WEBHOOK_URL = os.getenv(
    "WECOM_WEBHOOK",
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY_HERE",
)

# ---------- 定时报告（美东时间） ----------
# 盘前 1 小时 / 盘后 1 小时
SCHEDULE_TIMES_ET = ["08:30", "17:00"]

# ---------- 每日报告配置 ----------
DAILY_REPORT_LIMIT = 10
DAILY_PRESELECT_LIMIT = 20

# ---------- 期权筛选参数 ----------
# 只抓距今天 MIN_DTE ~ MAX_DTE 天内到期的合约
MIN_DTE = 7
MAX_DTE = 60

# IV 高/低百分位阈值（用于标注信号）
IV_HIGH_PERCENTILE = 75   # IV > 75th percentile -> 偏高
IV_LOW_PERCENTILE  = 25   # IV < 25th percentile -> 偏低

# 只展示 OTM 比例在此范围内的合约（正负百分比）
OTM_RANGE_PCT = 0.15      # +/-15% 虚值范围

# ---------- 无风险利率（用于 Greeks 计算） ----------
RISK_FREE_RATE = 0.05     # 5% 年化

# ---------- 报告输出 ----------
REPORT_DIR = "reports"    # 本地报告保存目录
