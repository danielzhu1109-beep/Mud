# 🖥️ 智能期权分析器 · 终端版

完整复刻并升级自小红书截图的交易终端 UI，前后端分离。

---

## 📁 文件结构

```
options_terminal/
├── index.html        ← 前端界面（黑绿终端风）
├── app.py            ← Flask 后端 API（所有计算在这里）
├── requirements.txt
└── README.md
```

---

## 🚀 三步启动

### 第一步：安装依赖
```bash
pip install -r requirements.txt
```

### 第二步：启动后端
```bash
python app.py
```
看到 `Running on http://0.0.0.0:5000` 即成功。

### 第三步：打开前端
直接在浏览器打开：
```
http://127.0.0.1:5000
```
或者双击 `index.html`（需保证 app.py 正在运行）。

---

## ✨ 功能对照原图

| 原图功能 | 本版实现 |
|---------|---------|
| 自然语言扫描指令 | ✅ 文本框 + 历史记录 |
| Ticker 覆盖 / AI Provider | ✅ 下拉选择 |
| 启用AI / 三个诸葛亮开关 | ✅ 可切换 Toggle |
| 技术偏向卡片 | ✅ SMA20/50 + RSI 自动计算 |
| 现价参考 | ✅ 实时拉取 |
| 日内 VWAP | ✅ 5分钟分时自动计算 |
| AI模式/IV Rank | ✅ IV Rank 自动计算 |
| 最终方案策略文本 | ✅ 自动生成结构化建议 |
| 日K收盘路径图 | ✅ Canvas 渐变折线图 |
| 分时 + VWAP图 | ✅ 双线（价格+VWAP虚线） |
| 期权候选池（8张卡片） | ✅ 含 Greeks、IV、OI、成交量 |
| Longbridge 鉴权区 | ✅ UI 完整，可接入真实 SDK |
| AI Provider 管理 | ✅ 可增删 |

---

## ⌨️ 快捷键
- `Ctrl+Enter` / `Cmd+Enter` → 立即扫描

---

## 🔧 自定义

**修改分析标的**：在前端 Ticker 输入框直接修改

**修改 DTE 范围**：前端 DTE 范围输入框调整（5~30 天）

**修改方向**：CALL / PUT / 双向 下拉

**接入真实 AI**：在 `app.py` 的 `_strategy_text()` 函数中集成 DeepSeek / Qwen API，替换现有规则生成逻辑

**接入 Longbridge SDK**：在 `app.py` 中替换 yfinance 的数据源为 `longbridge` Python SDK

---

## 📡 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/scan` | POST | 期权扫描，返回合约+策略+Greeks |
| `/api/chart` | GET | 日K + 分时数据 |
| `/api/vwap_pct` | GET | 日内 VWAP 偏离百分比 |

---

_⚡ 数据来源：Yahoo Finance · 仅供学习参考，不构成投资建议_
