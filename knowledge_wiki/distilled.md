# Strategy Distillate

- updated_at: 2026-06-03T21:10:04.981004-04:00
- confidence: 0.56
- raw_summary: 系统当前更偏 双向，DTE 偏好 balanced，IV 偏好 balanced，风险收益偏好 mid，真实平仓样本 9 笔，开放模拟持仓 1 笔（有效样本权重 0.32），自主跟踪信号 0 条，学习置信度 0.56。

## Essence
- 系统当前更偏 双向，DTE 偏好 balanced，IV 偏好 balanced，风险收益偏好 mid，真实平仓样本 9 笔，开放模拟持仓 1 笔（有效样本权重 0.32），自主跟踪信号 0 条，学习置信度 0.56。
- 近7天方向偏好: balanced；近30天方向偏好: call；谨慎来源: scan, pool, unusual；规避环境: sentiment_neutral, valuation_fair, trend_bullish；规避因子: ivrv_neutral, skew_neutral, flow_weak
- 近7天 8 笔样本 / 0 条信号，方向偏好 balanced
- 研究强势股: TSLA

## Playbook
- none

## Avoid
- 谨慎来源: scan, pool, unusual
- 规避因子: ivrv_neutral, skew_neutral, flow_weak
- 近期平仓反馈显示 wide spread 持续拖累结果，自动仓继续回避。
- 近期亏损更多来自 pool 来源，自动仓已下调其优先级。

## Focus Symbols
- COST: 样本 PnL 424.0 / 胜率 51.1%
- AAPL: mega-cap liquidity, AI/device cycle, and clean short-DTE expression
- ORCL: enterprise AI/cloud re-rating plus repeatable trend structure
- RTX: 样本 PnL -324.0 / 胜率 0.0%
- TSLA: 历史研究 quality -1.97 / confidence 0.359

## Feedback Summary
- 近期亏损更多来自来源 pool x8
- 近期亏损更多来自 liquidity wide x9
- 近期主要平仓原因 bulk_exit_all x10

## Recent Feedback
- 2026-06-02T05:23:34.606559-04:00: UNH CALL loss pnl=-99.85% | 降低 CALL / short DTE 的优先级；规避 weak flow | next=下一次优先缩短或拉长 DTE 到更匹配样本胜率的区间，并缩短持有周期。
- 2026-06-02T05:23:36.420326-04:00: ORCL PUT loss pnl=-99.24% | 降低 PUT / short DTE 的优先级；规避 wide spread；规避 weak flow | next=下一次等正股重新站回触发位，或者直接切换到反向结构，不要原方向硬扛。
- 2026-06-02T05:23:38.045120-04:00: RTX PUT loss pnl=-94.74% | 降低 PUT / short DTE 的优先级；规避 wide spread；规避 weak flow | next=下一次优先缩短或拉长 DTE 到更匹配样本胜率的区间，并缩短持有周期。
- 2026-06-01T11:24:29.334135-04:00: UNH CALL loss pnl=-99.85% | 降低 CALL / short DTE 的优先级；规避 weak flow | next=下一次优先只保留自动仓；如果再手动介入，只挑 setup_confidence 更高、流动性更好的单子。
- 2026-06-01T11:24:29.334135-04:00: ORCL PUT loss pnl=-99.24% | 降低 PUT / short DTE 的优先级；规避 wide spread；规避 weak flow | next=下一次优先只保留自动仓；如果再手动介入，只挑 setup_confidence 更高、流动性更好的单子。
- 2026-06-01T11:24:29.334135-04:00: RTX PUT loss pnl=-94.74% | 降低 PUT / short DTE 的优先级；规避 wide spread；规避 weak flow | next=下一次优先只保留自动仓；如果再手动介入，只挑 setup_confidence 更高、流动性更好的单子。
- 2026-06-01T11:24:17.187819-04:00: UNH CALL loss pnl=-99.85% | 降低 CALL / short DTE 的优先级；规避 weak flow | next=下一次优先只保留自动仓；如果再手动介入，只挑 setup_confidence 更高、流动性更好的单子。
- 2026-06-01T11:24:17.187819-04:00: ORCL PUT loss pnl=-99.24% | 降低 PUT / short DTE 的优先级；规避 wide spread；规避 weak flow | next=下一次优先只保留自动仓；如果再手动介入，只挑 setup_confidence 更高、流动性更好的单子。
- 2026-06-01T11:24:17.187819-04:00: RTX PUT loss pnl=-94.74% | 降低 PUT / short DTE 的优先级；规避 wide spread；规避 weak flow | next=下一次优先只保留自动仓；如果再手动介入，只挑 setup_confidence 更高、流动性更好的单子。
- 2026-06-01T04:00:00.423118-04:00: UNH CALL loss pnl=-99.85% | 降低 CALL / short DTE 的优先级；规避 weak flow | next=下一次优先只保留自动仓；如果再手动介入，只挑 setup_confidence 更高、流动性更好的单子。

## Market Heat
- none

## Official Context
- none