# P-004 拥挤衰减检测 — 影子转正 SOP（P-20260826-004 落地）

> 状态: 判据已写入 `scripts/calibrate_crowding_decay.py`，2026-08-26 起生效。
> 当前影子模式运行中（历史快照 4/20）。

## 一、转正决策流程

```
每日 Phase 4 执行 --daily 归档
        │
        ▼
历史快照 ≥ 20 交易日
        │
        ▼
python scripts/calibrate_crowding_decay.py --calibrate
        │
        ├─ 网格回放: 5 corr × 6 drop × 3 confirm = 90 组合 → F1 最优
        │
        ▼
转正判据三条件 (全满足才 alert_enabled=True)
```

## 二、转正判据（三条件 AND）

| 判据 | 标准 | 不满足时 |
|---|---|---|
| (a) 阈值合理性 | 最优 corr ∈ [0.15, 0.50] 且 drop ∈ [0.15, 0.50] | 维持影子再观察 10 交易日 |
| (b) 校准质量 | F1 ≥ 0.25 且 precision ≥ 0.45 且 TP ≥ 5 | 同上 |
| (c) 事件回放 | 2026-07 行业回撤事件（锚点 07-15）前 ≥5 交易日，至少一族多空波动率比 >1.5 连续 2 日触发 | 同上 |

事件回放实现（`_replay_event_lead`）:
- 每族取首个可求值代表因子，用 daily_prices.csv 回填 2026-06-20~08-10 的日度 ls_vol_ratio
- 数据窗口前置 150 自然日供 rolling(60/120) 预热
- 2026-08-26 实测: 波动率适应族 2026-06-22 首次触发，lead = 16.4 交易日 ✅

## 三、转正后的告警处置动作（SOP）

检测器触发「族拥挤衰减」告警（corr>阈值 且 IC 环比降幅>阈值 连续 confirm 天）后：

1. **自动（零干预原则保持）**: 仅记录 + 日报风险预警节显式输出
2. **该族新因子降权**: 该范式新因子在 S5 门禁按告警强度折扣（需另行提案，不在本 SOP 范围内）
3. **人工决策点**: 日报列出告警族 → 用户决定是否暂停该族种子注入 / 降低实盘权重
4. **禁止**: 任何自动减仓 / 自动清仓动作（实盘动作必须人工）

## 四、回滚路径

- 判据未通过 → `crowding_decay_config.json` 写 `alert_enabled: false` + `promotion_check` 详情
- 转正后误报率恶化（周度复核 F1 跌破 0.2）→ 手动置 `alert_enabled: false` 回影子模式
- 检测算法主体（decay_monitor.scan_crowding_decay）本 SOP 未改动

## 五、关键常量位置

`scripts/calibrate_crowding_decay.py`:
- `PROMOTION_CORR_RANGE / PROMOTION_DROP_RANGE / PROMOTION_MIN_F1 / PROMOTION_MIN_PREC / PROMOTION_MIN_TP / PROMOTION_EVENT_LEAD`
- `EVENT_ANCHOR = '2026-07-15'`（中证网/好买 2026-07 行业回撤报道窗口，可随事件认知更新）
