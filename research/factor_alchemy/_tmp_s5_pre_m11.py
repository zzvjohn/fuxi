# -*- coding: utf-8 -*-
"""
S5 Joint Filter — 真实数据联合正向过滤 (年度超额 + Calmar)
=============================================================

基于中金 CICC Loop Engineering 的"11项联合过滤"思想实现:
  - 要求因子在两个独立年度 (2025, 2026) 的超额收益同时为正
  - Calmar 比率 > 1.0 (年化收益 / 最大回撤)
  - 使用真实日频行情数据 (Tushare daily_prices.csv + CSI 500 index)

计算流程:
  1. 加载日频行情数据 (open/high/low/close/volume/amount)
  2. 对每个候选因子，评估其公式在面板数据上的原始信号值
  3. 周频调仓: 每周最后一个交易日按信号排名，选 Top-N 做多 (等权)
  4. 计算组合日收益率 vs 中证500基准
  5. 分年度 (2025, 2026) 统计超额收益和 Calmar 比率
  6. 输出 S5 过滤结果

集成路径:
  - multi_stage_validator.py S5 阶段: 替代硬编码默认值
  - Ralph Loop E 阶段后调用, 为 JQ 提交做最终过滤

用法:
    from s5_joint_filter import S5JointFilter
    
    s5 = S5JointFilter(top_n=80)
    result = s5.validate_factor("overnight_momentum", "(open_p/close_p.shift(1)-1).rolling(5).mean()")
    print(result.passed, result.excess_2025, result.excess_2026, result.calmar_2025)
"""

import sys
import ast
import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# P-20260911-margin (2026-09-11): 两融字段 (Tushare 2026-08 起 10 列, 新增后 3 列入库)
_MARGIN_COLS = ["rzye", "rqye", "rzmre", "rqyl", "rqchl", "rzche", "rqmcl", "rzrqye"]


# 2026-09-18 (P-20260918-022): 任一 pandas 调用带 axis=1 的正则判据。
# 容忍任意空白与参数顺序 (`rank(axis=1, pct=True)` / `rank(pct=True,axis=1)` /
# `rank(pct=True, ascending=False, axis=1)` 全命中), 用于补齐字面白名单的漏判。
_AXIS1_CALL_RE = re.compile(r'\.\s*\w+\s*\([^()]*\baxis\s*=\s*1\b')


def _s5_holdings_mark() -> str:
    mode = (os.environ.get('FUXI_S5_HOLDINGS_MARK', 'enforce') or 'enforce').strip().lower()
    if mode not in ('legacy', 'enforce'):
        raise ValueError('FUXI_S5_HOLDINGS_MARK must be legacy or enforce')
    return mode


def _lhb_tri_on() -> bool:
    """🔴 P-20260918-001 单一真源: 龙虎榜**席位三路**通道 (top_inst) 是否加载。

    off = 修正前行为 (S5 只有 top_list 4 个汇总字段, 三路席位公式 NameError)。
    """
    raw = str(os.environ.get("FUXI_LHB_TRI_CHANNEL", "on") or "on").strip().lower()
    return raw not in ("off", "0", "false", "no", "disable", "disabled", "none")


# 九张三路席位字段 (inst/north/yz 各 3), 供 _build_series_context 注入
_LHB_TRI_COLS = [
    "lhb_inst_net_buy", "lhb_inst_buy", "lhb_inst_sell",
    "lhb_north_net_buy", "lhb_north_buy", "lhb_north_sell",
    "lhb_yz_net_buy", "lhb_yz_buy", "lhb_yz_sell",
]


# ═══════════════════════════════════════════════════════════════════════════
# 外源加载器因子通道 (v0.7.19, P-20260915-012, 2026-09-15)
# ---------------------------------------------------------------------------
# 断点: NOVEL_FACTOR_LIBRARY 里有 6 条「裸标识符公式」(3 条 lion_* + 3 条
# kuaidao_*), 形如 {'formula': 'lion_bias'}。这类条目的值由 data_loader_ext
# 的专用加载器直接产出宽表, **不是 pandas 表达式**, 因此:
#   1. _check_unknown_fields 把它们当未知字段 → 仅 WARNING (不阻断);
#   2. _check_formula_sanity 规则6 数 pandas 操作数 = 0 < 2 →
#      「过度简化」硬拦截, 在回测前就返回失败。
# 结果: 这 6 条因子在 S5 里 100% 假失败 (失败原因与因子质量无关)。
# 本通道把裸标识符直接路由到对应加载器, 绕过公式层求值与合理性拦截。
#   on  : 裸标识符 → 加载器 (默认)
#   off : 修复前行为 (仍被「过度简化」拦截), 零回归
# 注意: 本通道只解决「求值可达性」, 不改变 S5 裁决口径 (_verdict 未动)。
# ═══════════════════════════════════════════════════════════════════════════
LOADER_FORMULA_REGISTRY = {
    # slot 语义按加载器返回类型区分 (v0.7.20):
    #   int  → 加载器返回元组/list, 按位置下标取
    #   str  → 加载器返回 dict {字段名: 宽表}, 按字段名取
    # data_loader_ext.load_lion_anchor_wide → 元组 (index=trade_date, columns=ts_code)
    "lion_bias": ("load_lion_anchor_wide", 0),
    "lion_lu_days": ("load_lion_anchor_wide", 1),
    "lion_lu_cnt": ("load_lion_anchor_wide", 2),
    # data_loader_ext.load_kuaidao_fund_wide → dict {字段名: 日频宽表}
    #   注: kuaidao_dv_yield (公式 dv_ttm) 不走本表 —— dv_ttm 是 S5 可加载的
    #   普通字段, 它的阻断点是「裸字段被 _check_formula_sanity 判过度简化」,
    #   属另一类修复 (裸单字段白名单), 见报告 §6 P1 备注。
    "kuaidao_insider_net_buy": ("load_kuaidao_fund_wide", "kd_holder_net_ratio_3m"),
    "kuaidao_top10_ratio": ("load_kuaidao_fund_wide", "kd_top10_hold_ratio"),
}


def _loader_formula_switch() -> str:
    v = (os.environ.get("FUXI_S5_LOADER_FORMULA", "on") or "on").strip().lower()
    return v if v in ("on", "off") else "on"


def _parse_yyyymmdd(s: pd.Series) -> pd.Series:
    """纯整数 YYYYMMDD → datetime。

    🔴 必须显式 ``%Y%m%d``: pandas 3.x 下 int YYYYMMDD 配 ``format='mixed'``
    会走 epoch 误解析 (2026-08-31 实证)。兜底再试一次宽松解析 (兼容脏值)。
    """
    _s = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out = pd.to_datetime(_s, format="%Y%m%d", errors="coerce")
    _bad = out.isna() & _s.notna() & (_s != "")
    if _bad.any():
        out.loc[_bad] = pd.to_datetime(_s[_bad], errors="coerce")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# P0 引擎口径修正开关组 (v0.7.23, 2026-09-15)
# ---------------------------------------------------------------------------
# 背景: 2026-09-15 的 S5 口径复核确认 backtest_factor 存在 6 处会让**读数
# 失真**的实现问题 (详见 reports/2026-09-15_lion_s5_verdict.md §12)。
# 这些问题不改变 S5 的判据 (_verdict 未动), 只影响喂给判据的数字。
#
# 🔴 默认值分两批 (零回归保证移到 A/B 台架的 LEGACY 配置):
#    第 1 批 **已翻默认** (v0.7.26): MIN_COVERAGE 0→0.9 / TIE_RULE legacy→on /
#      COST_BPS 0→15 —— 三项均属「只删不加 / 单向收紧 / 去任意性」,
#      对裁决只会 PASS→FAIL, 不会凭空造出 PASS (判据见 P0 影响说明 §8.1)。
#    第 2 批 **已翻默认** (v0.7.27): TRADABLE_GATE off→on + ADJ_PRICE off→on,
#      **捆绑同批** (用户 2026-09-16 决策)。捆绑理由: TRADABLE_GATE 只删不加,
#      但 ADJ_PRICE 是**注入型** —— 单独翻会让"建仓日封涨停/买不进"那段收益
#      凭空进入读数, 与已记的 lion_lu_days 假 PASS 同型; 两者同批时该段收益
#      被闸门剔除 ⇒ 不产生假 PASS。
#      ⚠️ 残余风险 (已立提案 P-20260916-010): ADJ_PRICE 恢复的正是
#      「建仓日收盘→次日收盘」那段, 而 moneyflow 通道是**同日对齐** merge
#      (T 日盘后披露贴到 T 日行) ⇒ 对引用 moneyflow 的因子, 前视从「存在」
#      变为「可兑现」。修法是给该通道补 open(T+1) 位移, 见 P-20260916-010。
#    FIRSTDAY 保持 **off** —— 已被 ADJ_PRICE 取代的 dead path
#      (L1641 硬编码 ``_lo_op = ">" if _has_pre else (">=" if _first else ">")``
#       ⇒ ADJ_PRICE 生效时 FIRSTDAY 的 ">=" 分支不可达, 两者恢复的是同一天)。
#    逐格边际影响见 research/s5_p0_ablation.py, 基线 GOLDEN_prechange.json。
#    复现修正前读数: 显式设全部 6 个 env 为旧值 (= 该台架的 LEGACY 配置,
#      即 TRADABLE_GATE=off FIRSTDAY=off TIE_RULE=legacy COST_BPS=0
#      MIN_COVERAGE=0 ADJ_PRICE=off, 应与 GOLDEN 逐位一致)。
#      ⚠️ v0.7.26/v0.7.27 起「不设 env = 旧行为」**不再成立**, 必须显式设回。
# ═══════════════════════════════════════════════════════════════════════════

def _s5_tradable_gate() -> bool:
    """P0-a 可成交闸门。默认 ``on`` (v0.7.27 翻默认; 旧默认 off)。

    缺陷: 建仓候选**不过滤涨停** ⇒ 信号日在涨停板封死的票照样被选入,
    次日无法买入, 收益是"纸面收益"。实测 lion_lu_days(−) 有 **80.7%** 的
    建仓日标的处于涨停封板状态 (报告 §5.1), 其 "+81.93%/IR 3.90" 式读数
    全部由买不进的票贡献。

    口径: 精确法用 ``stk_limit.up_limit`` (close_p >= up_limit 视为封板买不进);
    缺 limit 行的日期 (实测数据至 2026-08-03) 走兜底
    ``close_p/pre_close_p - 1 >= 板块幅度 - 0.5pp``, 板块按 ts_code 前缀判定
    (主板 10% / 创业板·科创板 20% / 北交所 30%)。
    注意: **只拦建仓侧 (买不进)**。持有期内的跌停无法卖出会顺延到下一期,
    本引擎不建模"顺延持仓", 故只在诊断里计数 (n_blocked_limit_down) 以标记
    该期收益偏乐观, 不做拦截。
    """
    v = (os.environ.get("FUXI_S5_TRADABLE_GATE", "on") or "on").strip().lower()
    return v in ("on", "1", "true", "yes", "enforce")


def _s5_first_day() -> bool:
    """P0-b 含持有期首日。默认 off (**故意保持**, 已被 ADJ_PRICE 取代)。

    缺陷 (L1405 原式 ``trade_date > entry_date``): 切片从建仓日**次日**开始,
    而 ``pct_change()`` 在该切片首行必然 NaN ⇒ **每期固定丢掉一个交易日的
    收益**。实测 288 期中 265 期的建仓日落在周一, 而 2025 周一等权收益
    +38.17bp vs 其他日 +7.39bp ⇒ 全市场等权 2025 读数 +35.45% → +8.05%
    (**−27.40pp**)。零分布佐证: 2025 超额 0/400 为正 (报告 §9)。

    🔴 为何永远是 off (v0.7.27 复核): 本项与 ADJ_PRICE **不是互补而是替代** ——
    两者恢复的是**同一天** (建仓日收盘→次日收盘)。建仓切片运算符硬编码为
    ``_lo_op = ">" if _has_pre else (">=" if _first else ">")`` (L1641),
    ADJ_PRICE 生效 ⇒ ``_has_pre`` 为真 ⇒ FIRSTDAY 的 ``">="`` 分支**不可达**。
    故 ADJ_PRICE 翻 on 后本项是纯 dead path, 翻它不会带来额外修正。
    留档 / 消融对照时才置 on。
    """
    v = (os.environ.get("FUXI_S5_FIRSTDAY", "off") or "off").strip().lower()
    return v in ("on", "1", "true", "yes", "enforce")


def _s5_tie_rule() -> str:
    """P0-c/d 并列处理口径。默认 ``on`` (v0.7.26 翻默认; 旧默认 legacy)。

    - ``legacy``: ``sort_values(ascending=False)`` = pandas 默认 quicksort,
      并列块内顺序**未定义且依赖输入行序** ⇒ 同一因子两次运行可能选出不同
      组合 (不可复现)。
    - ``stable``: 只把排序换成 mergesort (**P0-d 单项**) — 顺序确定但仍
      **任意** (等于输入 ts_code 序)。实测单独生效会让裁决翻转
      (+4.81%→−0.61%), 故**不建议单独使用**。
    - ``on``     : **P0-c+d 同批** — 主键 factor_value 降序, 并列块内再按
      ``amount`` 降序 (流动性优先: 同等信号强度选可承接的票), 尾部以
      ``ts_code`` 升序做最终确定性打散键。三条键全确定性 ⇒ 可复现。
      依据: 库内并列>80 的期占 45.7%; 而库内离散型因子仅 ≈3.1% (8/259),
      故影响面**窄但真实**。
    """
    v = (os.environ.get("FUXI_S5_TIE_RULE", "on") or "on").strip().lower()
    return v if v in ("legacy", "stable", "on") else "on"


def _s5_cost_bps() -> float:
    """P0-g 双边交易成本 (基点)。默认 15 (v0.7.26 翻默认; 旧默认 0)。

    缺陷: 引擎对 ``commission/fee/slippage`` **零命中**, S5 收益是**费前无
    滑点**数字。实现: 每期按实际换手 ``2 × |新\\旧| / top_n`` 计费, 在持有
    期首日一次性扣除。取值参考: A股双边 ≈ 佣金 3bp + 印花税 5bp (卖出) +
    冲击成本 ⇒ 15~30bp 为常见敏感性区间。
    """
    try:
        return max(0.0, float(os.environ.get("FUXI_S5_COST_BPS", "15")))
    except (TypeError, ValueError):
        return 15.0


def _s5_return_alignment() -> bool:
    mode = os.environ.get('FUXI_S5_RETURN_ALIGN', 'on').strip().lower()
    if mode not in ('on', 'off'):
        raise ValueError('FUXI_S5_RETURN_ALIGN must be on or off')
    return mode == 'on'


def _s5_min_coverage() -> float:
    """P0-h 调仓期覆盖率下限。默认 0.9 = 覆盖不足判不通过 (v0.7.26 翻默认)。

    缺陷 (L1395 ``if len(entry_data) < self.top_n: continue``): 覆盖不足的
    调仓周被**静默整周丢弃** — 不记录、不计入, 且 ``ann_ret = total*PPY/len``
    的自适应年化会掩盖少算的期数 ⇒ 因子"只在有覆盖的周里表现好"即可刷高读数
    (实测 lion_bias 截面覆盖最小值 = 1)。设 >0 时: 覆盖率低于该值则**判定
    不通过**, 原因写明。
    """
    try:
        v = float(os.environ.get("FUXI_S5_MIN_COVERAGE", "0.9"))
    except (TypeError, ValueError):
        return 0.9
    return v if 0.0 <= v <= 1.0 else 0.9


def _s5_adj_price() -> str:
    """P1 除权调整收益口径。默认 ``on`` (v0.7.27 翻默认; 旧默认 off)。
    返回模式串: off | raw | on。

    🔴 **注入型开关, 默认翻 on 的处理已定 (2026-09-16)**: 本项单独翻会凭空
    造出假 PASS (它恢复的正是"建仓日封涨停/买不进"的那段收益) ⇒ 用户决策为
    **与 TRADABLE_GATE 捆绑同批翻默认**, 由闸门把买不进的标的剔除后再注入。
    残余前视风险与修法见文件头开关区块注释 + 提案 P-20260916-010。

    缺陷: ``close_p.pct_change()`` 用**未复权** close ⇒ 除权除息日 (送转/高
    分红) 的除权缺口被记成伪负收益。正确口径 ``close/pre_close - 1``
    (``pre_close`` 已由数据源按公司行为调整)。

    ⚠️ **三档的必要性 (2026-09-15 实测补记)**:
      - ``off`` : 修正前口径 ``pct_change()``。
      - ``raw`` : ``close/pre_close - 1`` **无守卫**。实测会出现
        ``pre_close`` = 发行价的新股首日 ⇒ 单日 +200400% 级伪收益
        (报告 §12.3 已记录该形态)。此档**仅用于留证/消融对照**。
      - ``on``  : 同 raw **但加板块限幅守卫** —— |r| 超过该股票板块涨跌停
        幅度 +0.5pp 的行置 NaN (丢弃)。这是**推荐档**: 既修掉除权伪负收益,
        又不让无涨跌停限制的新股/复牌首日伪收益进入组合。
    数据自洽性已审计 (`research/p1_preclose_audit.py`): 99.654% 行的
    ``pre_close`` 恒等于前收, 偏离的 0.346% 全部为除权调整 (方向 23900↓/38↑)
    ⇒ ``pre_close`` 与 ``close`` 同源同口径, 可作收益分母。

    实测低估量 (共同支撑集口径, ``research/s5_unadj_final.py``):
    2025 **+3.84pp** / 2026 **+2.63pp** (截尾稳健下界 +2.19/+1.47pp);
    除权日原始 −4.203% vs 正确 +0.291%。

    🔴 **对 S5 裁决的真实位移 (2026-09-15 消融实测, 比上面大一个量级)**:
        lion_bias(+) 2025 超额 −19.75% (off) → +3.94% (P0_b) → **+15.20% (raw)**;
        ctrl 因子 tot25 +20.81% (off) → +44.35% (P0_b) → +45.17% (raw)。
        P1 − P0_b 的 **+11.26pp** 已做污染归因 (`research/p1_outlier_contamination.py`):
          · 组合内 |r_pre| 最大的 8 个标的**全部是 2021–2022 年的 920xxx.BJ**
            (新股首日 pre_close=发行价), **不在 2025/2026 裁决窗口内**;
          · 丢弃全部 |r_pre|>30% 的 45 行后, lion_bias(+) 2025 仅 −2.35pp、
            2026 0.00pp ⇒ 污染只解释 ~2.35pp;
          · 余下 ~+8.9pp 属**真实除权修正**, 与市场整体口径差 (+8.4pp/yr, 审计 [6])
            量级吻合 ⇒ **P1 的 PASS 不是伪影**, 加守卫后仍成立。
    ⚠️ 不复权 ``adj_factor.csv`` (季频快照, 不可日频用)。
    """
    v = (os.environ.get("FUXI_S5_ADJ_PRICE", "on") or "on").strip().lower()
    if v in ("on", "enforce", "guard"):
        return "on"
    if v in ("raw", "1", "true", "yes"):
        return "raw"
    return "off"


def _s5_idx_align() -> str:
    """P-20260917-021 **基准同窗裁剪**开关 (根因修复, 默认 ``off`` 待 A/B)。

    🔴 缺陷 (2026-09-17 实测, 详见 ``data/baseline_misalign_20260917.json``):
      · 价格 df 被裁到 ``_ceil_ts`` (L~910), 但基准 ``idx_df`` 在 L~951
        是 ``pd.read_csv(index_path)`` —— **从不裁剪**;
      · ``compute_metrics`` 里 ``port_year`` 与 ``idx_year`` 虽然用同一个
        ``_end``, 但两侧 df 的**可用末端不同** ⇒ ``excess = total_return −
        benchmark_return`` 的两端**不同期**;
      · 实测旧默认 ``ceiling=2026-08-28``: 组合止于 08-28、基准取到 09-09,
        多含的 8 个交易日指数 **−0.61%** ⇒ 基准被低估 **0.73pp** ⇒ 超额被
        **高估 +0.73pp** ⇒ 系统性指向假 PASS (即"偏松")。
      · 该偏差对每条因子是**同一常数** (基准全池共用, 组合逐条不同) ⇒ 它
        平移全池的 ex26 中心, 不改变散布。

    ⚠️ 为何默认 off 而不是直接修:
      裁剪后基准会**去掉**那段行情 —— 该段涨则该修正**放宽**、跌则**收紧**,
      方向**取决于行情**而非固定单向 ⇒ 不满足"只删不加/单向收紧"的可安全
      翻默认判据 ⇒ 须走 A/B (同 60 条、同 caliber、只切本开关)。

    与 ``FUXI_S5_DATE_CEIL=auto`` 的关系 (易混, 务必分清):
      · ``auto`` 是**取窗**策略: 把窗口定在必需通道交集末端 ⇒ 让价格与基准
        **在当前采集顺序下碰巧末端相同**。指数落后于价格时它顺带对齐了,
        但指数**领先**时 (采集顺序反转) 错配会重现。
      · 本开关是**对齐**本身: 无论窗口定在哪, 基准都被裁到与价格同一个
        ``_ceil_ts`` ⇒ 与采集顺序无关。
      ⇒ 两者不是重复: auto 治症, 本开关治根。当前数据下 (指数落后) 二者
        读数等价, 故 auto 的 A/B 已顺带覆盖了本开关的当前效果。
    """
    v = (os.environ.get("FUXI_S5_IDX_ALIGN", "off") or "off").strip().lower()
    return "on" if v in ("on", "1", "true", "yes", "align") else "off"


_AUX_LAG_MAX = 60


def _s5_aux_lag() -> int:
    """P-20260916-010 **辅助通道交易日滞后**开关 (默认 ``off`` = 0 交易日)。

    🔴 要解决的问题 (**选股侧 1 日前视**, 2026-09-16 立项):
      ``_load_auxiliary_data`` 把 ``moneyflow_daily`` / ``margin_detail`` **同日**
      merge 进价格 df ⇒ 在 T 日的公式里 ``net_mf_amount`` / ``rzye`` 取的是
      **T 日盘后披露**的值, 而组合在 T 日收盘建仓 ⇒ 存在 1 日前视。
      (moneyflow 为 T 日盘后公布; 两融明细同为 T 日盘后。)

    ⚠️ 前视**未放大**实证 (立项时): 同日 vs 滞后 1 日在既有候选池上
      ``p`` = 0.33 / 0.59 ⇒ 差异不显著。故本开关**默认 off**, 且**不允许**
      拿"不显著"当翻默认的理由 —— 它治的是**口径正确性**, 不是收益率。

    ``on`` 语义 = 每只股票**自身交易日序列**上的 ``shift(N)`` (N 个**交易日**,
      不是自然日, 不用 ``DateOffset``)。缺失保持缺失; moneyflow 侧保留既有
      ``fillna(0)`` 约定, 但顺序为 **先移位后填充** (反之会把"无数据→0"
      搬到 T 日污染真值)。

    ``_p`` 后缀列与主列**同步移位** (二者本为同一个值, 只移一侧 = 静默错配)。

    ⚠️ 与"缺失→0"既有约定的交互 (**已知, 本批不改**): moneyflow 原本就把缺失日
      填 0 ⇒ 移位后该 0 会落到**次日**。这与"缺失日读 0"是同一约定的两个时点;
      单变量 A/B 只动**时点**, 不动缺失处理 (稀疏导致的 0 传播属既有口径问题,
      ``FUXI_MF_MISSING=nan`` 在 ``data_loader_ext`` 侧另治)。

    与 ``FUXI_S5_DATE_CEIL`` (取窗) / ``FUXI_S5_IDX_ALIGN`` (基准同窗) 正交。

    支持: ``off``(0, 默认) / ``on``(=1) / 显式整数 (上限 60, 防误设巨大值
    静默掏空整条通道)。非法值一律回落 0。
    """
    v = (os.environ.get("FUXI_S5_AUX_LAG", "off") or "off").strip().lower()
    if v in ("", "off", "0", "false", "no", "none", "disable", "disabled"):
        return 0
    if v in ("on", "true", "yes", "enable", "enabled"):
        return 1
    try:
        n = int(float(v))
    except Exception:
        return 0
    return max(0, min(n, _AUX_LAG_MAX))


def _shift_aux_trading_multi(df, cols, lag: int) -> None:
    """P-20260916-010: 把 ``df[cols]`` 在每只股票**自身交易日序列**上后移 lag 行。

    实现 (只改值, 不打乱行序, 不新增持久列):
      1. 取 ``(ts_code, trade_date)`` 键帧 + 各列副本;
      2. 按 ``(ts_code, trade_date)`` 排序 ⇒ ``groupby("ts_code").shift(lag)``
         (行序内后移 = **交易日**滞后, 无需自然日历);
      3. ``sort_index()`` 复原原行序 ⇒ 按位置回填。

    🔴 **必须批量**: 逐列调用 = 18 次全表排序 + 18 份键帧副本 (7M 行量级)。
    缺失保持缺失 (不隐式填 0 或前向填充)。

    ``lag <= 0`` 或无有效列 → 空操作 (零回归)。
    """
    _cols = [c for c in (cols or []) if c in df.columns]
    if int(lag or 0) <= 0 or not _cols:
        return
    t = df[["ts_code", "trade_date"]].copy()
    for _c in _cols:
        t[_c] = df[_c].values
    t = t.sort_values(["ts_code", "trade_date"], kind="mergesort")
    _g = t.groupby("ts_code", sort=False)
    for _c in _cols:
        t[_c] = _g[_c].shift(int(lag))
    t = t.sort_index(kind="mergesort")
    for _c in _cols:
        df[_c] = t[_c].values


def _shift_aux_trading(df, col: str, lag: int) -> None:
    """单列薄封装 (便于单测直达; 生产路径走 ``_shift_aux_trading_multi`` 批量版)。"""
    _shift_aux_trading_multi(df, [col], lag)


def _s5_ledger_mode() -> str:
    """M11: S5 账本口径开关 ``FUXI_S5_LEDGER`` (默认 ``proxy``)。

    ``proxy``        — 历史口径: 周频换成员 / 日频等权收益 (改动前逐位一致);
    ``fixed_shares`` — 固定股数现金账本 (``portfolio_execution_ledger``,
                       声明模型情景; 未闭合 ⇒ data_insufficient, 缺数可枚举;
                       未决陈旧 mark 不授 valid)。

    非法值 fail-fast (不静默回落)。校验逻辑单一来源 = ``s5_ledger_bridge.
    ledger_mode``; bridge 不可导入时本地兜底同义实现 (两处需同步)。
    """
    try:
        from s5_ledger_bridge import ledger_mode
        return ledger_mode()
    except ImportError:
        raw = (os.environ.get("FUXI_S5_LEDGER", "proxy") or "proxy").strip().lower()
        if raw not in ("proxy", "fixed_shares"):
            raise ValueError(
                "FUXI_S5_LEDGER must be proxy|fixed_shares, got %r" % raw)
        return raw


# M11 测试/隔离注入点: 账本 bars 的 raw 目录覆盖 (None = data/raw 生产默认)。
_LEDGER_RAW_DIR: Optional[Path] = None


def _s5_sparse_diag() -> str:
    """P-20260916-014a **稀疏结构体检**开关 (纯披露, 零裁决)。默认 ``on``。

    🔴 要解决的问题 (2026-09-16 实证): `crowded_amount_reversal_5d` 在 **FA 层
    ICIR=0.693 显著为正**, 但 S5 全量池 TOP80 组合两口径均强负超额
    (legacy 零成本 ex25 −25.08% / ex26 −6.40%; default ex25 −31.39% / ex26 −8.90%)。
    整个流程**没有任何检查**会发现这类「IC 正 / TOP-N 反」的错位 —— 因子照样
    跑完、照样给读数, 失败归因只能靠人工复刻 85 期才做得出。

    ⚠️ **归因已被两次修正, 最终结论**:
      (1) 初判「表示错配」(TOP-N 混入 `value==0` 未触发股, 实测占 36.7%)
          —— **被分组检验否证**: corr(0 值占比, 期超额) = **+0.066 ≈ 0**;
          且 0 值占比 ==0 的 27 期期均超额 **−0.4108%**, 反而**差于**混入 0 值股的
          期 (−0.0714%) ⇒ 排掉 0 值股组合**更差**, 改 NaN 救不活。
      (2) 真病灶 = **尾部失效**: 85 期平均十分位曲线 argmax 落在 **Q9**
          (+0.673%/期), Q10 回落至 +0.446%, 仅 **6/9** 递增对 (非单调)。
          S5 的 TOP-N 正是从 Q10 内部再取最极端 N 只 ⇒ **在已失效区取极端值**。
    ⇒ 「argmax 是否落在 Q10」比「0 值占比」更本质, 故本体检**两项都查**。

    披露字段 (写入 ``S5Result.sparse_diag`` + 命中时追加到 ``reason``):
      · ``sel_zero_frac_mean`` / ``sel_nan_frac_mean`` : TOP-N 中未触发占比
      · ``n_pos`` (触发且值为正的股数) 与 ``pct_periods_pos_below_topn``
      · ``decile_curve_mean`` / ``decile_argmax`` / ``decile_monotone_pairs``
      · ``flags`` : topn_zero_frac_high | argmax_not_top_decile |
                    decile_non_monotone | pos_below_topn

    阈值 (命中即告警, 均**不影响裁决**):
      · TOP-N 0 值占比 > 0.20
      · argmax ≠ 第 10 分位 (末位)
      · 递增对 < 8/9
      · 有效信号数 < top_n 的期数占比 > 0.5

    成本: 逐期多一次 `isin((entry, exit))` 扫 + 一次 pivot (实测 85 期量级
    的增量在秒级)。``FUXI_S5_SPARSE_DIAG=off`` 可完全关闭 ⇒ 逐位复现旧读数。
    """
    v = (os.environ.get("FUXI_S5_SPARSE_DIAG", "on") or "on").strip().lower()
    return v if v in ("on", "off") else "on"


SPARSE_ZERO_FRAC_WARN = 0.20      # TOP-N 中 value==0 占比告警线
SPARSE_MONO_MIN_PAIRS = 8         # 十分位递增对下限 (满 9)
SPARSE_POS_BELOW_TOPN_WARN = 0.5  # 「有效信号数 < top_n」的期数占比告警线
SPARSE_DECILES = 10


def _decile_curve_period(entry_df: pd.DataFrame, fwd_ret: pd.Series,
                         n_dec: int = SPARSE_DECILES):
    """单期十分位持有期收益曲线 (P-20260916-014a)。

    按 ``factor_value`` 的横截面百分比位强制分 ``n_dec`` 组 (重复值用
    ``method="first"`` 打散, **不用** ``qcut`` —— 稀疏因子大量并列 0 会让
    qcut 的 bin 边界塌缩), 返回各组等权持有期收益均值。

    Args:
      entry_df: 建仓日截面, 需含 ``ts_code`` / ``factor_value``
      fwd_ret : 持有期收益, ``index=ts_code`` (建仓价→下期建仓价)
    Returns:
      list[float] 长度 ``n_dec`` (缺组为 NaN), 或 None (不可算)
    """
    try:
        if entry_df is None or len(entry_df) == 0:
            return None
        if fwd_ret is None or len(fwd_ret) == 0:
            return None
        if "factor_value" not in entry_df.columns:
            return None
        _f = pd.to_numeric(entry_df["factor_value"], errors="coerce")
        _ok = _f.notna().to_numpy()
        if int(_ok.sum()) < n_dec * 2:
            return None
        _codes = entry_df["ts_code"].to_numpy()[_ok]
        _rr = fwd_ret.reindex(_codes).to_numpy(dtype="float64")
        if not np.isfinite(_rr).any():
            return None
        _vals = _f.to_numpy()[_ok]
        _rk = pd.Series(_vals).rank(pct=True, method="first").to_numpy()
        _dec = np.minimum((_rk * n_dec).astype(int), n_dec - 1)
        _df = pd.DataFrame({"dec": _dec, "r": _rr}).dropna()
        if len(_df) == 0:
            return None
        _g = _df.groupby("dec")["r"].mean()
        return [float(_g.get(i, np.nan)) for i in range(n_dec)]
    except Exception:  # noqa: BLE001  纯披露: 任何异常都不得影响裁决
        return None


def _sparse_mean_curve(rows, n_dec: int = SPARSE_DECILES):
    """逐期曲线列表 → 逐分位均值 (全 NaN 列返回 NaN, 屏蔽 nanmean 告警)。"""
    _a = np.array([r for r in (rows or []) if r], dtype=float)
    if _a.size == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(_a, axis=0)


def _sparse_curve_stats(m):
    """十分位均值曲线 → (argmax 1-based, 递增对数, Q末−Q末前)。"""
    if m is None or not np.isfinite(m).any():
        return None, None, None
    _pairs = int(sum(1 for i in range(len(m) - 1)
                     if np.isfinite(m[i]) and np.isfinite(m[i + 1])
                     and m[i + 1] > m[i]))
    _d = (None if not (np.isfinite(m[-1]) and np.isfinite(m[-2]))
          else round(float(m[-1] - m[-2]), 6))
    return int(np.nanargmax(m)) + 1, _pairs, _d


def summarize_sparse_diag(periods, top_n: int = 80,
                          zero_frac_warn: float = SPARSE_ZERO_FRAC_WARN,
                          mono_min: int = SPARSE_MONO_MIN_PAIRS) -> Optional[Dict]:
    """把逐期稀疏统计汇总成一次『稀疏结构体检』(P-20260916-014a, 纯披露)。

    纯函数 (不读 env / 不碰 self), 便于单测与复算。

    🔴 **为何必须分窗** (2026-09-16 实测): 全窗(2021 起, 285 期)十分位均值曲线的
    argmax = **Q5**, 而**裁决窗**(2025-2026, 85 期)的 argmax = **Q9**。S5 只对
    2025/2026 两年做裁决 ⇒ 用全窗 argmax 判定「尾部是否失效」会**指错方向**
    (全窗混入 2021-2024 的历史结构)。故:
      · ``decile_curve_mean``   : 全窗 (参考)
      · ``decile_curve_recent`` : 裁决窗 (``FUXI_S5_SPARSE_RECENT_FROM``,
        默认 ``2025-01-01``) —— **flags 以此为准**
      · ``decile_by_year``      : 逐年曲线 + argmax (定位是哪一年失效)
    """
    _ps = [p for p in (periods or []) if isinstance(p, dict)]
    if not _ps:
        return None
    n = len(_ps)
    _npos = np.array([p.get("n_pos", 0) or 0 for p in _ps], dtype=float)
    _nsig = np.array([p.get("n_signal", 0) or 0 for p in _ps], dtype=float)
    _zfr = np.array([p.get("sel_zero_frac", 0.0) or 0.0 for p in _ps], dtype=float)
    _nfr = np.array([p.get("sel_nan_frac", 0.0) or 0.0 for p in _ps], dtype=float)
    out = {
        "n_periods": n,
        "top_n": int(top_n),
        "n_pos": {"min": float(np.min(_npos)), "p25": float(np.percentile(_npos, 25)),
                  "median": float(np.median(_npos)),
                  "p75": float(np.percentile(_npos, 75)),
                  "max": float(np.max(_npos)), "mean": float(np.mean(_npos))},
        "n_signal": {"min": float(np.min(_nsig)), "median": float(np.median(_nsig)),
                     "mean": float(np.mean(_nsig))},
        "pct_periods_pos_below_topn": float((_npos < int(top_n)).mean()),
        "pct_periods_signal_below_topn": float((_nsig < int(top_n)).mean()),
        "sel_zero_frac_mean": float(np.mean(_zfr)),
        "sel_nan_frac_mean": float(np.mean(_nfr)),
        "sel_zero_frac_max": float(np.max(_zfr)),
    }

    flags = []
    _recent_from = (os.environ.get("FUXI_S5_SPARSE_RECENT_FROM",
                                   "2025-01-01") or "2025-01-01").strip()
    out["recent_from"] = _recent_from

    # 全窗曲线 (参考)
    _m_all = _sparse_mean_curve([p.get("decile") for p in _ps])
    _am_all, _pa_all, _d_all = _sparse_curve_stats(_m_all)
    if _m_all is not None:
        out["decile_curve_mean"] = [None if not np.isfinite(v) else round(float(v), 6)
                                    for v in _m_all]
        out["decile_argmax"] = _am_all
        out["decile_argmin"] = int(np.nanargmin(_m_all)) + 1
        out["decile_monotone_pairs"] = "%d/%d" % (_pa_all, len(_m_all) - 1)
        out["decile_monotone_n"] = _pa_all
        out["decile_top_minus_prev"] = _d_all
        out["n_periods_with_curve"] = int(sum(1 for p in _ps if p.get("decile")))

    # 裁决窗曲线 (flags 以此为据)
    _rec_all = [p for p in _ps if str(p.get("date", ""))[:10] >= _recent_from]
    _rec_ps = [p for p in _rec_all if p.get("decile")]
    _m_rec = _sparse_mean_curve([p.get("decile") for p in _rec_ps])
    _am_rec, _pa_rec, _d_rec = _sparse_curve_stats(_m_rec)
    if _m_rec is not None:
        out["decile_curve_recent"] = [None if not np.isfinite(v) else round(float(v), 6)
                                      for v in _m_rec]
        out["decile_argmax_recent"] = _am_rec
        out["decile_monotone_pairs_recent"] = "%d/%d" % (_pa_rec, len(_m_rec) - 1)
        out["decile_monotone_n_recent"] = _pa_rec
        out["decile_top_minus_prev_recent"] = _d_rec
        out["n_periods_recent"] = len(_rec_ps)

    # 裁决窗**标量** (与曲线同窗): 全窗均值会被 2021-2024 的结构稀释
    # (实测 crowded: 全窗 sel_zero_frac 33.2% vs 裁决窗 36.7%) ⇒ 半套分窗
    # 会留下同样的误导, 故标量也一并分窗, flags/warning 全用裁决窗。
    if _rec_all:
        _zr_r = np.array([p.get("sel_zero_frac", 0.0) or 0.0 for p in _rec_all],
                         dtype=float)
        _nfr_r = np.array([p.get("sel_nan_frac", 0.0) or 0.0 for p in _rec_all],
                          dtype=float)
        _npos_r = np.array([p.get("n_pos", 0) or 0 for p in _rec_all], dtype=float)
        out["sel_zero_frac_mean_recent"] = float(np.mean(_zr_r))
        out["sel_nan_frac_mean_recent"] = float(np.mean(_nfr_r))
        out["n_pos_recent"] = {"min": float(np.min(_npos_r)),
                               "median": float(np.median(_npos_r)),
                               "mean": float(np.mean(_npos_r))}
        out["pct_periods_pos_below_topn_recent"] = float((_npos_r < int(top_n)).mean())
        out["n_periods_recent_all"] = len(_rec_all)

    # 逐年曲线 (定位失效年份)
    _by_year = {}
    for _y in sorted({str(p.get("date", ""))[:4] for p in _ps}):
        if not _y.isdigit():
            continue
        _rows = [p.get("decile") for p in _ps
                 if str(p.get("date", ""))[:4] == _y and p.get("decile")]
        _m = _sparse_mean_curve(_rows)
        _am, _pa, _d = _sparse_curve_stats(_m)
        if _m is None:
            continue
        _by_year[_y] = {
            "n_periods": len(_rows),
            "curve": [None if not np.isfinite(v) else round(float(v), 6) for v in _m],
            "argmax": _am,
            "monotone_pairs": "%d/%d" % (_pa, len(_m) - 1),
            "top_minus_prev": _d,
        }
    if _by_year:
        out["decile_by_year"] = _by_year

    # ── flags: 一律以**裁决窗**为准 (无裁决窗数据时回退全窗) ──
    _am_use = _am_rec if _m_rec is not None else _am_all
    _pa_use = _pa_rec if _m_rec is not None else _pa_all
    _am_src = "recent" if _m_rec is not None else "full"
    _zfr_use = out.get("sel_zero_frac_mean_recent",
                       out["sel_zero_frac_mean"])
    _posb_use = out.get("pct_periods_pos_below_topn_recent",
                        out["pct_periods_pos_below_topn"])
    out["decile_flag_basis"] = _am_src
    if _am_use is not None:
        if _am_use != SPARSE_DECILES:
            flags.append("argmax_not_top_decile")
        if _pa_use < mono_min:
            flags.append("decile_non_monotone")
    if _zfr_use > zero_frac_warn:
        flags.append("topn_zero_frac_high")
    if _posb_use > SPARSE_POS_BELOW_TOPN_WARN:
        flags.append("pos_below_topn")

    _ws = []
    if "topn_zero_frac_high" in flags:
        _ws.append("TOP-N 未触发(0值)占 %.1f%%" % (_zfr_use * 100))
    if _am_use is not None and "argmax_not_top_decile" in flags:
        _ws.append("十分位 argmax=Q%d(非 Q10, %s窗)"
                   % (_am_use, "裁决" if _am_src == "recent" else "全"))
    if _pa_use is not None and "decile_non_monotone" in flags:
        _ws.append("单调性 %d/%d" % (_pa_use, SPARSE_DECILES - 1))
    if "pos_below_topn" in flags:
        _ws.append("有效信号数<top_n 的期占 %.0f%%" % (_posb_use * 100))
    out["flags"] = flags
    out["warning"] = ("⚠️ 稀疏结构: " + " / ".join(_ws)
                      + " ⇒ 尾部可能失效 (IC 有效 ≠ TOP-N 有效), 送 JQ 前需查十分位曲线"
                      ) if _ws else ""
    return out


# 板块涨跌停幅度 (P0-a 兜底判定用, 精确法 stk_limit 缺失时)
def _board_limit_vector(codes: pd.Series) -> pd.Series:
    """v0.7.23 P0-a: 按 ts_code 前缀向量化返回涨跌停幅度 (小数)。

    主板 10% / 创业板·科创板 (30x·68x) 20% / 北交所 (.BJ 或 4·8 开头) 30%。
    """
    s = codes.astype(str)
    lim = pd.Series(0.10, index=s.index, dtype="float64")
    _bj = s.str.endswith(".BJ") | s.str[:1].isin(["4", "8"])
    lim[_bj] = 0.30
    _gem = s.str[:2].isin(["30", "68"])
    lim[_gem & ~_bj] = 0.20
    return lim


def _parse_multiline_formula(formula: str):
    """v0.7: 多行公式解析 — 返回 (body_lines, final_expr)。

    单行公式返回 (None, formula)。
    多行: 丢弃注释/空行, 前 N-1 行为赋值体 (保留原始缩进, 整块 exec), 末行剥离赋值为表达式。
    末行为缩进块的一部分时返回 (全部行, None) — 调用方整块 exec 后取 factor/result 变量。
    """
    raw_lines = [ln for ln in formula.split("\n")
                 if ln.strip() and not ln.strip().startswith("#")]
    if len(raw_lines) <= 1:
        return None, formula
    stripped = [ln.strip() for ln in raw_lines]
    final_raw = raw_lines[-1]
    # 末行缩进 → 属于上一语句块 (for/if 体), 需整块 exec
    if final_raw[0].isspace():
        return raw_lines, None
    body = raw_lines[:-1]
    final = stripped[-1]
    m = re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$', final)
    if m:
        final = m.group(1).strip()
    return body, final


def infer_periods_per_year(index) -> int:
    """v0.8: 从收益序列索引推断年化因子 (每年观测周期数)。

    基于相邻观测的中位时间间隔:
      - 中位间隔 <= 3 天 → 252 (日频, 容忍轻度缺数据)
      - 其他            → round(365.25 / 中位间隔)
        (7天→52 周频, 14天→26 双周, ~30.4天→12 月频, 91天→4 季度)
    无有效间隔/异常时回退 252 (默认日频, 保持历史口径)。
    """
    try:
        dt = pd.to_datetime(pd.Index(index))
        if len(dt) < 2:
            return 252
        diffs = dt.to_series().diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return 252
        median_days = float(diffs.median()) / 86400.0
    except Exception:
        return 252

    if median_days <= 3:
        return 252          # 日频
    return int(round(365.25 / median_days))


@dataclass
class S5Result:
    """S5 验证结果"""
    factor_name: str = ""
    formula: str = ""
    passed: bool = False
    reason: str = ""
    
    # 年度超额收益 (vs 中证500)
    excess_2025: float = 0.0
    excess_2026: float = 0.0
    
    # Calmar 比率
    calmar_2025: float = 0.0
    calmar_2026: float = 0.0
    
    # 组合统计
    total_return_2025: float = 0.0
    total_return_2026: float = 0.0
    benchmark_return_2025: float = 0.0
    benchmark_return_2026: float = 0.0
    max_drawdown_2025: float = 0.0
    max_drawdown_2026: float = 0.0
    annual_vol_2025: float = 0.0
    annual_vol_2026: float = 0.0

    # ── P-20260912-003 Calmar 口径可比性修正 ─────────────────────────
    # 2025 是完整年度(~245 交易日), 2026 仅 01~09(~180 交易日)。
    # maxDD 随窗口长度单调增大 ⇒ 同阈值比较 c25>1 与 c26>1 属「苹果比橘子」,
    # 2026 的 Calmar 被系统性高估。裁决口径由 FUXI_S5_CALMAR_MODE 控制。
    calmar_mode: str = "legacy"          # legacy | common_window | vol_scaled
    calmar_2025_full: float = 0.0        # 2025 全长年度 Calmar (诊断对照)
    calmar_2026_full: float = 0.0
    window_days_2025: int = 0            # 实际参与计算 2025 的交易日数
    window_days_2026: int = 0
    calmar_window_matched: Any = None    # 两窗口交易日数是否已对齐
    calmar_threshold_effective: float = 1.0   # 当前口径下实际生效的阈值

    # ── v0.7.23 P0 口径披露字段 (零裁决, 纯留痕) ─────────────────────
    # P0-h: 覆盖不足而被静默跳过的调仓期此前完全不可见 ⇒ 读数无法归因。
    #       覆盖率 = 实际参与期数 / 可能期数; 设 FUXI_S5_MIN_COVERAGE>0 时
    #       低于该值会**判定不通过** (见 validate_factor)。
    coverage_rate: float = 1.0              # 调仓期覆盖率 (1.0 = 无跳过)
    periods_skipped: int = 0                # 被跳过的调仓期数
    coverage_ok: bool = True                # 覆盖率是否达 FUXI_S5_MIN_COVERAGE
    engine_switches: Any = None             # 本次回测生效的 P0/P1 开关快照

    # ── P-20260916-014a 稀疏结构体检 (零裁决, 纯披露) ────────────────
    # 「IC 显著为正但 TOP-N 组合反向」这一失败模式此前**完全不可见** (只能靠
    # 人工复刻逐期数据才发现)。字段含: TOP-N 未触发占比 / 有效信号数 vs top_n /
    # 十分位曲线 + argmax 位置 + 单调递增对数 / flags。命中 flag 时在 reason
    # 末尾追加 ⚠️ 注记, **不改 passed**。FUXI_S5_SPARSE_DIAG=off 关闭。
    sparse_diag: Any = None

    # 计算状态
    error: str = ""
    computation_time: float = 0.0
    marking_diagnostics: Any = None
    economic_readings_status: str = "not_evaluated"

    # v0.6.1: 组合收益序列 (内存对象, 不序列化; S6 DSR + 增量门禁的数据通道)
    portfolio_returns: Any = None
    portfolio_returns_n: int = 0

    # ── P-20260910-008 周频链路修复: 分频口径审计字段 ──────────────
    # 执行层此前只有日频数据通道 → natural_freq=="weekly" 的公式在 S5 被
    # 按日频宽表求值 (rolling(4) 从 4 周变 4 日), 语义被破坏。
    signal_freq: str = "daily"          # 本次裁决实际使用的口径
    weekly_channel_mode: str = "off"    # off | shadow | enforce | fallback_daily
    weekly_ic: float = 0.0              # 周频面板 Rank IC (与 S1 对拍用)
    weekly_icir: float = 0.0
    weekly_n: int = 0
    weekly_excess_2025: float = 0.0     # 周频口径分年超额
    weekly_excess_2026: float = 0.0
    weekly_calmar_2025: float = 0.0
    weekly_calmar_2026: float = 0.0
    weekly_passed: Any = None           # 周频口径通过的裁决
    weekly_reason: str = ""
    daily_caliber_passed: Any = None    # shadow 模式: 日频口径裁决 (现行行为对照)
    daily_caliber_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "passed": self.passed,
            "reason": self.reason,
            "excess_2025": round(self.excess_2025, 4),
            "excess_2026": round(self.excess_2026, 4),
            "calmar_2025": round(self.calmar_2025, 2),
            "calmar_2026": round(self.calmar_2026, 2),
            "total_return_2025": round(self.total_return_2025, 4),
            "total_return_2026": round(self.total_return_2026, 4),
            "max_drawdown_2025": round(self.max_drawdown_2025, 4),
            "max_drawdown_2026": round(self.max_drawdown_2026, 4),
            "error": self.error,
            "marking_diagnostics": self.marking_diagnostics,
            "economic_readings_status": self.economic_readings_status,
            # P-20260910-008
            "signal_freq": self.signal_freq,
            "weekly_channel_mode": self.weekly_channel_mode,
            "weekly_ic": round(self.weekly_ic, 6),
            "weekly_icir": round(self.weekly_icir, 4),
            "weekly_n": self.weekly_n,
            "weekly_excess_2025": round(self.weekly_excess_2025, 4),
            "weekly_excess_2026": round(self.weekly_excess_2026, 4),
            "weekly_calmar_2025": round(self.weekly_calmar_2025, 2),
            "weekly_calmar_2026": round(self.weekly_calmar_2026, 2),
            "weekly_passed": self.weekly_passed,
            "weekly_reason": self.weekly_reason,
            "daily_caliber_passed": self.daily_caliber_passed,
            "daily_caliber_reason": self.daily_caliber_reason,
            # P-20260912-003 Calmar 口径可比性
            "calmar_mode": self.calmar_mode,
            "calmar_2025_full": round(self.calmar_2025_full, 2),
            "calmar_2026_full": round(self.calmar_2026_full, 2),
            "window_days_2025": self.window_days_2025,
            "window_days_2026": self.window_days_2026,
            "calmar_window_matched": self.calmar_window_matched,
            "calmar_threshold_effective": self.calmar_threshold_effective,
            # v0.7.23 P0 口径披露
            "coverage_rate": round(self.coverage_rate, 4),
            "periods_skipped": self.periods_skipped,
            "coverage_ok": self.coverage_ok,
            "engine_switches": self.engine_switches,
            # P-20260916-014a 稀疏结构体检
            "sparse_diag": self.sparse_diag,
        }


class S5JointFilter:
    """
    S5 联合正向过滤器: 基于真实行情数据的年度超额+Calmar验证。
    
    对标中金: 要求因子在两个独立年度同时产生正超额 + Calmar > 1.0。
    """

    def __init__(
        self,
        price_path: Path = None,
        index_path: Path = None,
        mcap_path: Path = None,
        top_n: int = 80,
        excess_threshold: float = 0.0,
        calmar_threshold: float = 1.0,
        sample_stocks: int = None,
        rebalance_freq: str = "W",  # Weekly
        cache_dir: Path = None,
    ):
        self.price_path = price_path or RAW_DIR / "daily_prices.csv"
        self.mcap_path = mcap_path or DATA_DIR / "jqdata" / "mcap_weekly.csv"
        self.top_n = top_n
        self.excess_threshold = excess_threshold
        self.calmar_threshold = calmar_threshold
        self.rebalance_freq = rebalance_freq
        self.cache_dir = cache_dir or DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── P-20260910-008 周频链路修复: 执行层周频数据通道 ──
        # 架构声明「执行层 S5/S6/JQ 唯一周频口径」(见 weekly_lane.py 头部),
        # 但实现只落在 S1: S5 只有日频长表通道 → weekly 候选的 rolling(N)
        # 被当作 N 日求值, 22/22 过 τ_w 的周频种子无一能过 S5。
        #   off     : 修复前行为 (周频候选仍走日频口径), 零回归
        #   shadow  : 周频口径照算并落盘审计, 裁决仍用日频口径 (兼容路径)
        #   enforce : 默认，natural_freq=="weekly" 按周频裁决，失败不得替换频率
        _wc = (os.environ.get("FUXI_S5_WEEKLY_CHANNEL", "enforce") or "enforce").strip().lower()
        if _wc not in ('off', 'shadow', 'enforce'):
            raise ValueError('FUXI_S5_WEEKLY_CHANNEL must be off, shadow or enforce')
        self.weekly_channel = _wc
        self.weekly_min_stocks = int(os.environ.get("FUXI_S5_WEEKLY_MIN_STOCKS", "30"))
        self._weekly_panel = None

        # ── P-20260909-A (2026-09-09): S5 全量池统一裁决 ──
        # 历史遗留: sample_stocks=200 按成交额取 Top-200 (大盘池, 仅覆盖全A
        # 成交额 25.9%), 而生成端持续产出中小盘 alpha 因子 → 结构性误杀
        # (桥接首跑 0/17 实证; 招商/东方证券研报: 量价因子小市值组显著更强)。
        # 现默认全量池 (None=不过滤); S1-S4 本就是全市场口径, 至此五项裁决
        # 统一。旧口径可经 FUXI_S5_UNIVERSE=<N> 显式回退 (Top-N 流动性池)。
        _env_universe = os.environ.get("FUXI_S5_UNIVERSE")
        if sample_stocks is not None:
            self.sample_stocks = int(sample_stocks)
        elif _env_universe is not None and str(_env_universe).strip() != "":
            self.sample_stocks = int(_env_universe)
        else:
            self.sample_stocks = None

        # ── P-20260909-A: 基准指数默认改中证全指 (000985, 与全量股票池对齐) ──
        # 旧基准 000905 中证500 与 Top-200 大盘池配套; 全量池用全指口径。
        # 若显式传入 index_path 则尊重调用方; 基准随池口径自动切换。
        if index_path is not None:
            self.index_path = index_path
        elif (self.sample_stocks is None
              and (RAW_DIR / "index_000985_daily.csv").exists()):
            self.index_path = RAW_DIR / "index_000985_daily.csv"
        else:
            self.index_path = RAW_DIR / "index_000905_daily.csv"

        # 延迟加载
        self._price_df: Optional[pd.DataFrame] = None
        self._index_df: Optional[pd.DataFrame] = None
        self._universe: Optional[List[str]] = None
        self._loaded = False
        # v0.7.23 P0-a: 涨跌停面板 (仅 FUXI_S5_TRADABLE_GATE=on 时惰性加载)
        self._limit_panel: Optional[pd.DataFrame] = None
        # v0.7.23 P0-h: 最近一次 backtest_factor 的覆盖率/成本诊断 (只披露不裁决)
        self._last_backtest_diag: Optional[Dict] = None

    @property
    def is_ready(self) -> bool:
        """检查数据是否可用"""
        return self.price_path.exists() and self.index_path.exists()

    def _load_data(self):
        """加载价格数据和基准数据"""
        if self._loaded:
            return

        print(f"  [S5] Loading price data from {self.price_path}...")
        # v0.9.3 内存修复 (2026-08-21): 只读 S5 必需列 (usecols), 并在选定
        # universe 后立刻裁剪, 避免 13.3M 行 × ~30 列 float64 在辅助数据
        # merge 时 consolidate 出 (25, 13303024) 单块 2.48GiB → _ArrayMemoryError。
        _read_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                      # v0.7.23 P1: pre_close 供除权调整收益口径 (close/pre_close-1)。
                      # 单列 float32 ≈ 28MB, 相对既有 7M×8 列可忽略; 不读 adj_factor
                      # (季频快照, 不可日频用)。
                      "pre_close"]
        df = pd.read_csv(
            self.price_path,
            usecols=[c for c in _read_cols if c in
                     pd.read_csv(self.price_path, nrows=0).columns],
            low_memory=False,
        )
        # 统一列名: daily_prices.csv 可能用 vol/amount 或其他名称
        col_map = {
            "vol": "volume", "turnover_vol": "volume",
            "amt": "amount", "turnover_val": "amount",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 确保必要列存在
        required = ["ts_code", "trade_date", "open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"    S5 ERROR: Missing columns: {missing}")
            df_cols = list(df.columns)
            print(f"    Available: {df_cols}")
            self._price_df = None
            return

        # 补全可选列
        for col in ["volume", "amount"]:
            if col not in df.columns:
                df[col] = 0  # 用0填充缺失列
        
        # 2026-08-31 数据修复: 裸 to_datetime 对纯 int YYYYMMDD 列按纳秒解析→1970年
        # (旧文件混格式侥幸读成 object 才没炸); 统一 astype(str)+mixed 双格式兼容。
        df["trade_date"] = pd.to_datetime(
            df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
            format="mixed", errors="coerce")
        df = df.dropna(subset=["trade_date"])
        
        # 过滤日期范围 (有足够数据)
        # 2026-08-31 数据修复: 上限 08-07 为 8/8 遗留硬编码 (当时数据只到 08-07);
        # 重采后数据至 08-28 (周五, 完整周), 对齐修复后数据末端。
        # 🔴 2026-09-16 P-20260916-013: 该上限是**硬编码常量**, 数据已推进到
        # 09-15 而此处未跟随 ⇒ S5 **静默**丢弃 12 个交易日 / 2 个完整调仓期。
        # 唯一症状是打印的 "Loaded N rows" 与 raw CSV 行数不符 (7,005,133 vs
        # 7,071,713) —— 不报错、不可见, 与「接线断点」同类。
        # 现两点改动:
        #   (a) 可配置: FUXI_S5_DATE_CEIL。**v0.7.34 起默认 "auto"** ——
        #       2026-09-17 翻默认, 前置 60 条全池 A/B 已过 (P-20260917-020):
        #       ① 无新增 PASS (两组各 0); ② 位移中心经「基准同窗错配修正」
        #       扣除后与 0 不可区分 (Bootstrap CI95 含 0) ⇒ 窗口延长本身无
        #       系统效应; ③ 残留散布 std=2.3pp 属个别因子的窗口敏感性;
        #       ④ 现行默认的错配方向是**高估超额** ⇒ 翻 auto 是收紧方向。
        #       旧默认 "2026-08-28" 仍完整可用: 显式设 env 即可; 且
        #       --caliber legacy 已显式挂回 08-28 ⇒ 历史零回归基线不受影响。
        #   (b) 让截断**可见**: 打印窗口 + 被丢弃的尾行数, 有截断时显式告警。
        _ceil = (os.environ.get("FUXI_S5_DATE_CEIL", "auto") or "auto").strip()
        _raw_max = df["trade_date"].max()
        _bind = "显式上限"
        if _ceil.lower() not in ("auto", "data", "none", ""):
            try:
                _ceil_ts = pd.Timestamp(_ceil)
            except Exception:
                print(f"    [S5] ⚠️ FUXI_S5_DATE_CEIL={_ceil!r} 解析失败, "
                      f"回退 2026-08-28")
                _ceil_ts, _bind = pd.Timestamp("2026-08-28"), "回退默认"
        else:
            # ── auto 语义 (P-20260916-017, 默认已翻 v0.7.34) ─────────────
            # 🔴 根因: 基准 df (L951 `idx_df = pd.read_csv(index_path)`)
            #    **从不裁剪**, 而价格 df 会被裁到 _ceil_ts ⇒ compute_metrics
            #    里 port_year 与 idx_year 虽用**同一 _end**, 两侧**可用末端却
            #    不同** ⇒ 超额的两端不同期 ⇒ 系统性污染。污染方向由「错配那
            #    段行情的涨跌」决定, 因此**双向**:
            #      · 组合多含 / 基准少含 (auto 取价格末端, 指数落后于价格):
            #        该段指数涨 ⇒ 基准被低估 ⇒ 超额虚高 ⇒ 假 PASS。
            #      · 组合少含 / 基准多含 (⚠️ 现行旧默认 ceil=08-28 就是这个
            #        形态: 价格被裁到 08-28 但指数仍取到 09-09): 实测该段指数
            #        −0.61% ⇒ 基准被低估 0.73pp ⇒ 超额**高估 +0.73pp** ⇒
            #        同样指向假 PASS (见 data/baseline_misalign_20260917.json)。
            #    ⇒ auto = S5 **裁决必需通道**的交集末端 (价格 ∩ 基准指数),
            #      让两侧末端**在当前采集顺序下**自动对齐。注意 auto 只是
            #      "顺带对齐", 并非根因修复 —— 采集顺序若反转 (指数领先价格)
            #      错配会重现 ⇒ 根因修复 = FUXI_S5_IDX_ALIGN (P-20260917-021,
            #      默认 off 待 A/B)。
            #    日频辅助通道 (moneyflow/margin) 缺失只让引用字段为 NaN、
            #    不改窗口 ⇒ 不参与取 min, 但一并披露末端供人工判断。
            #    故 auto = S5 **裁决必需通道**的交集末端 (价格 ∩ 基准指数)。
            #    日频辅助通道 (moneyflow/margin) 缺失只让引用字段为 NaN、
            #    不改窗口 ⇒ 不参与取 min, 但一并披露末端供人工判断。
            _ceil_ts = _raw_max
            _bind = "价格末端"
            _idx_max = None
            try:
                _ip = getattr(self, "index_path", None)
                if _ip is not None and Path(_ip).exists():
                    _idf = pd.read_csv(_ip)
                    _ic = next((c for c in _idf.columns if "date" in c.lower()), None)
                    if _ic is not None and len(_idf):
                        _idx_max = pd.to_datetime(
                            _idf[_ic].astype(str).str.replace(r"\.0$", "", regex=True),
                            format="mixed", errors="coerce").max()
            except Exception as _e:  # noqa: BLE001
                print(f"    [S5] ⚠️ auto 读基准指数末端失败, 退回价格末端: "
                      f"{str(_e)[:70]}")
            if _idx_max is not None and pd.Timestamp(_idx_max) < pd.Timestamp(_ceil_ts):
                _idx_ts = pd.Timestamp(_idx_max)
                _gap_days = int(df.loc[df["trade_date"] > _idx_ts, "trade_date"].nunique())
                _ceil_ts = _idx_ts
                _bind = (f"基准指数末端 (价格末端 {pd.Timestamp(_raw_max).date()},"
                         f" 差 {_gap_days} 个交易日)")
            self._auto_aux_tails = {}
            for _f, _n in ((RAW_DIR / "moneyflow_daily.csv", "moneyflow"),
                           (RAW_DIR / "margin_detail.csv", "margin")):
                try:
                    if _f.exists():
                        _ad = pd.read_csv(_f, usecols=lambda c: "date" in c.lower())
                        _ac = list(_ad.columns)[0]
                        self._auto_aux_tails[_n] = str(pd.to_datetime(
                            _ad[_ac].astype(str).str.replace(r"\.0$", "", regex=True),
                            format="mixed", errors="coerce").max().date())
                except Exception:  # noqa: BLE001
                    pass
        self._date_ceil = str(pd.Timestamp(_ceil_ts).date())
        self._date_ceil_binder = _bind
        _n_raw = len(df)
        _cut_dates = int(df.loc[df["trade_date"] > _ceil_ts, "trade_date"].nunique())
        df = df[(df["trade_date"] >= "2020-01-01") & (df["trade_date"] <= _ceil_ts)]
        _n_cut = _n_raw - len(df)
        if _n_cut:
            _aux = getattr(self, "_auto_aux_tails", None) or {}
            _aux_lag = [f"{k}={v}" for k, v in _aux.items() if v < self._date_ceil]
            print(f"    [S5] 日期窗上限 = {self._date_ceil} [{_bind}] | 数据末端 = "
                  f"{pd.Timestamp(_raw_max).date()} ⇒ 丢弃尾部 {_n_cut:,} 行"
                  f" ({_cut_dates} 个交易日)"
                  + (f" | 辅助通道落后于窗: {', '.join(_aux_lag)}" if _aux_lag else "")
                  + ("  ⚠️ 上限早于数据末端: S5 裁决窗口被截断"
                     " (FUXI_S5_DATE_CEIL=auto = 取必需通道交集末端)"
                     if _raw_max > _ceil_ts else ""))


        # v0.9.3: 数值列转 float32 (rank 相关/回测精度无影响, 内存减半)
        for c in ["open", "high", "low", "close", "volume", "amount", "pre_close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        
        # 创建 _p 后缀字段 (pandas 公式使用的字段)
        for col in ["open", "high", "low", "close", "volume", "amount", "pre_close"]:
            if col in df.columns:
                df[f"{col}_p"] = df[col]

        # 添加衍生字段
        df["overnight"] = df["open"] / df["close"].groupby(df["ts_code"]).shift(1) - 1
        df["overnight_p"] = df["overnight"]
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].groupby(df["ts_code"]).shift(1)
        df["amplitude_p"] = df["amplitude"]
        df["returns"] = df["close"].groupby(df["ts_code"]).pct_change()
        df["returns_p"] = df["returns"]
        df["turnover"] = df["volume"] / df["close"]
        df["turnover_p"] = df["turnover"]
        df["hl_ratio"] = df["high"] / df["low"]
        df["hl_ratio_p"] = df["hl_ratio"]

        self._price_df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        print(f"    Loaded {len(df)} rows, {df['ts_code'].nunique()} stocks")

        # 加载基准指数 (全量池 ⇒ 中证全指 000985)
        print(f"  [S5] Loading index data from {self.index_path}...")
        idx_df = pd.read_csv(self.index_path)
        idx_df["trade_date"] = pd.to_datetime(
            idx_df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
            format="mixed", errors="coerce")
        idx_df = idx_df.dropna(subset=["trade_date"]).sort_values("trade_date")
        # 🔴 P-20260917-021: 基准必须与组合**同窗**。旧实现此处**不裁剪** ⇒
        #    组合止于 _ceil_ts 而基准取到指数文件末端 ⇒ 超额两端不同期 ⇒
        #    系统性偏差 (实测旧默认下超额被高估 +0.73pp, 指向假 PASS)。
        #    ⚠️ 顺序: **先 pct_change 再裁**。反序会让裁剪后首日的 idx_return
        #    变成 NaN ⇒ 窗口内读数被改动 ⇒ 那就不是"纯对齐"而是改了口径。
        idx_df["idx_return"] = idx_df["close"].pct_change()
        _ia = _s5_idx_align()
        _idx_raw_n = len(idx_df)
        if _ia == "on":
            _cut = getattr(self, "_date_ceil", None)
            if _cut:
                idx_df = idx_df[idx_df["trade_date"] <= pd.Timestamp(_cut)]
        self._index_df = idx_df
        if _idx_raw_n != len(idx_df):
            print(f"    Loaded {len(idx_df)} index days [基准同窗裁剪 on: "
                  f"丢弃尾部 {_idx_raw_n - len(idx_df)} 行 ⇒ 与价格同末 "
                  f"{getattr(self, '_date_ceil', '?')}]")
        else:
            print(f"    Loaded {len(idx_df)} index days")

        # 选择 universe (采样流动性最好的 stocks)
        self._select_universe()

        # v0.9.3 内存修复: universe 选定后立即裁剪 price_df 到 universe 股票,
        # 再 merge 辅助数据。旧实现先 merge 后裁剪 → 13.3M 行 × 25 列
        # merge_asof 时 pandas consolidate 需一次性 (25, 13303024) float64
        # ≈ 2.48 GiB 连续内存, Windows 下必爆 (_ArrayMemoryError 2026-08-21 实证)。
        # P-20260909-A: 全量池 (sample_stocks=None) 时跳过裁剪 (isin 全 True
        # 的布尔过滤 + reset_index 在 7M 行上是纯浪费的整帧拷贝)。
        if self.sample_stocks is not None:
            self._price_df = self._price_df[
                self._price_df["ts_code"].isin(self._universe)
            ].reset_index(drop=True)

        # v0.6: 加载辅助数据 (moneyflow + balancesheet for P-008/P-009)
        self._load_auxiliary_data()

        self._loaded = True
        print(f"  [S5] Data loaded. Universe: {len(self._universe)} stocks")

    def _load_auxiliary_data(self):
        """v0.6: 加载 moneyflow 和 balancesheet 数据并 merge 到价格 DataFrame"""
        df = self._price_df
        if df is None:
            return

        # ── P-008: moneyflow_daily (主动资金流) ──
        mf_path = RAW_DIR / "moneyflow_daily.csv"
        mf_loaded = False
        if mf_path.exists():
            try:
                mf = pd.read_csv(mf_path, low_memory=False)
                # v0.5.4: trade_date 为 YYYYMMDD 数字格式, 需显式 format
                # (直接 to_datetime 会解析成 1970 epoch → merge 全部错位)
                try:
                    mf["trade_date"] = pd.to_datetime(
                        mf["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                except Exception:
                    mf["trade_date"] = pd.to_datetime(mf["trade_date"], errors='coerce')
                mf_cols = ["ts_code", "trade_date"]
                avail_mf = []
                # v0.6.2 (2026-08-29): 对齐 IC Computer 10 字段 (补 net_mf_amount 等,
                # 修复 seed v3_margin_mf_divergence eval 报 NameError: net_mf_amount)
                for col in ["buy_lg_vol", "sell_lg_vol", "buy_sm_vol", "sell_sm_vol",
                            "buy_md_vol", "sell_md_vol", "buy_elg_vol", "sell_elg_vol",
                            "net_mf_vol", "net_mf_amount"]:
                    if col in mf.columns:
                        mf_cols.append(col)
                        avail_mf.append(col)
                if len(avail_mf) > 0:
                    mf = mf[mf_cols].dropna(subset=["trade_date"])
                    # P-20260909-A 内存修复: 全量池 7M 行下整帧 merge 触发 pandas
                    # consolidate 大块连续内存分配 (2.48GiB OOM 根因复现风险)。
                    # 改 keys-only 对齐 + 逐列注入 (df[col]=m[col].values),
                    # 每次仅物化单列, 内存峰值 = 基础帧 + 2列。
                    m = df[["ts_code", "trade_date"]].merge(mf, on=["ts_code", "trade_date"], how="left")
                    # P-20260916-010: 辅助通道交易日滞后 (默认 0 = 零回归)。
                    # 顺序: 批量赋值 → 批量移位 → fillna(0) + 建 _p。
                    # 🔴 先移位后填 0 —— 反之会把"无数据→0"搬到 T 日污染真值。
                    _alag = _s5_aux_lag()
                    for col in avail_mf:
                        df[col] = m[col].values
                    _shift_aux_trading_multi(df, avail_mf, _alag)
                    for col in avail_mf:
                        df[col] = df[col].fillna(0)
                        df[f"{col}_p"] = df[col]
                    del m
                    mf_loaded = True
                    print(f"  [S5] Loaded moneyflow: {len(avail_mf)} fields ({', '.join(avail_mf)})"
                          + (f" [aux_lag={_alag}交易日]" if _alag else ""))
                else:
                    print(f"  [S5] ⚠️ moneyflow_daily.csv 无大单字段, 跳过")
            except Exception as e:
                print(f"  [S5] ⚠️ moneyflow load failed: {e}")
        else:
            print(f"  [S5] ⚠️ moneyflow_daily.csv not found")

        # ── v0.6.2 (2026-08-29): 两融数据 (margin_detail.csv) ──
        # 修复 seed v3_margin_mf_divergence eval 报 NameError: rzye is not defined
        # (S5 此前完全没有 margin 数据通道; IC Computer 有 5 字段)
        mg_path = RAW_DIR / "margin_detail.csv"
        if mg_path.exists():
            try:
                mg = pd.read_csv(mg_path, low_memory=False)
                mg["trade_date"] = pd.to_datetime(
                    mg["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                mg_cols = ["ts_code", "trade_date"] + [
                    c for c in _MARGIN_COLS
                    if c in mg.columns
                ]
                if len(mg_cols) > 2:
                    mg = mg[mg_cols].dropna(subset=["trade_date"])
                    # P-20260909-A: keys-only 逐列注入 (同 moneyflow, 防 7M 行 consolidate OOM)
                    m = df[["ts_code", "trade_date"]].merge(mg, on=["ts_code", "trade_date"], how="left")
                    # P-20260916-010: 辅助通道交易日滞后 (默认 0 = 零回归)
                    _alag = _s5_aux_lag()
                    _mg_ok = [c for c in _MARGIN_COLS if c in m.columns]
                    for col in _mg_ok:
                        df[col] = m[col].values
                    _shift_aux_trading_multi(df, _mg_ok, _alag)
                    for col in _mg_ok:
                        df[f"{col}_p"] = df[col]
                    del m
                    print(f"  [S5] Loaded margin: "
                          f"{len([c for c in _MARGIN_COLS if c in df.columns])} fields "
                          f"(至 {mg['trade_date'].max().date()})"
                          + (f" [aux_lag={_alag}交易日]" if _alag else ""))
            except Exception as e:
                print(f"  [S5] ⚠️ margin load failed: {e}")
        else:
            print(f"  [S5] ⚠️ margin_detail.csv not found")

        # ── P-009: balancesheet (无形资产/商誉) ──
        bs_path = RAW_DIR / "balancesheet.csv"  # 需先采集
        # 降级: 尝试从本地 tushare 缓存加载
        if not bs_path.exists():
            bs_path = RAW_DIR / "fina_indicator.csv"  # 用 fina 做降级

        if bs_path.exists() and "balancesheet" in str(bs_path):
            try:
                bs = pd.read_csv(bs_path, low_memory=False)
                bs_cols = ["ts_code", "end_date"]
                avail_bs = []
                for col in ["intan_assets", "goodwill", "total_assets"]:
                    if col in bs.columns:
                        bs_cols.append(col)
                        avail_bs.append(col)
                    elif col == "total_assets" and "total_assets" not in bs.columns:
                        # 尝试在 fina_indicator 中找替代
                        pass

                if len(avail_bs) > 0:
                    bs = bs[bs_cols].dropna(subset=["end_date"])
                    bs["end_date"] = pd.to_datetime(bs["end_date"], errors="coerce").astype("datetime64[ns]")
                    bs = bs.dropna(subset=["end_date"])
                    bs = bs.sort_values(["ts_code", "end_date"])

                    # 将季度财务数据 forward-fill 到每个交易日
                    # 策略: 每个 ts_code 的每个 trade_date 取最近的 end_date 数据
                    df_sorted = df.sort_values(["ts_code", "trade_date"]).copy()
                    # P-018 fix: 统一 datetime64 精度为 [ns] (merge_asof 要求两端一致)
                    df_sorted["trade_date"] = pd.to_datetime(df_sorted["trade_date"]).astype("datetime64[ns]")
                    # 使用 merge_asof 做前向填充
                    bs_sorted = bs.sort_values("end_date").rename(columns={"end_date": "trade_date"})
                    bs_sorted["trade_date"] = pd.to_datetime(bs_sorted["trade_date"]).astype("datetime64[ns]")

                    # merge_asof: 对每个 ts_code, 将最近的历史财务数据对齐到交易日
                    # P-20260909-A 内存修复: merge_asof 输出=左帧×全列, 全量池 7M 行
                    # × 40+ 列 consolidate 峰值 >4GB。改 keys-only asof: 左帧只带
                    # (ts_code, trade_date, _k), asof 后按 _k 还原行序逐列注入。
                    # (merge_asof 要求左键有序 → argsort 记录映射, 输出 roundtrip
                    #  还原; 单测验证 NaN/跨股/多期对齐正确)
                    bs_sorted = bs_sorted.sort_values("trade_date")
                    _left = df_sorted[["ts_code", "trade_date"]].reset_index(drop=True)
                    _left["_k"] = np.arange(len(_left))
                    _order = _left["trade_date"].values.argsort(kind="stable")
                    _sorted_left = _left.iloc[_order].reset_index(drop=True)
                    _merged = pd.merge_asof(
                        _sorted_left, bs_sorted,
                        by="ts_code", on="trade_date",
                        direction="backward",
                        tolerance=pd.Timedelta(days=180),
                    ).set_index("_k").sort_index()
                    # 对齐安全: df 与 df_sorted 同为 (ts_code,trade_date) 唯一键排序
                    # → 行序必然一致; 防御性验证 (不一致则按索引对齐)
                    _aligned_ok = (
                        (df.index == df_sorted.index).all()
                        and (_merged.index == np.arange(len(df))).all()
                    )
                    if _aligned_ok:
                        _vals = {col: _merged[col].values for col in avail_bs}
                    else:
                        _mi = pd.Series(np.arange(len(df)), index=df_sorted.index)
                        _vals = {col: _merged[col].reindex(_mi.reindex(df.index).values).values
                                 for col in avail_bs}
                    for col in avail_bs:
                        df[col] = _vals[col]
                        df[col] = df[col].fillna(0)
                        df[f"{col}_p"] = df[col]
                    del _left, _order, _sorted_left, _merged, _vals
                    bs_loaded = True
                    print(f"  [S5] Loaded balancesheet: {len(avail_bs)} fields ({', '.join(avail_bs)}) [asof merge, tolerance=180d]")
                else:
                    print(f"  [S5] ⚠️ No BS fields found, skip")
            except Exception as e:
                print(f"  [S5] ⚠️ balancesheet load failed: {e}")
        elif bs_path.exists():
            print(f"  [S5] ⚠️ No balancesheet.csv, fina_indicator has no intan/goodwill — P-009 templates will fail S5")
        else:
            print(f"  [S5] ⚠️ No financial data files — P-009 templates will fail S5")

        # ── v0.10 (2026-08-25 红利审计): 估值字段 (daily_basic 慢变量) ──
        basic_path = RAW_DIR / "daily_basic.csv"
        if basic_path.exists():
            try:
                _val_cols = ["pe_ttm", "pb", "dv_ratio"]
                ba = pd.read_csv(
                    basic_path,
                    usecols=[c for c in ["ts_code", "trade_date"] + _val_cols
                             if c in pd.read_csv(basic_path, nrows=0).columns],
                    low_memory=False)
                ba["trade_date"] = pd.to_datetime(
                    ba["trade_date"].astype(str), format='mixed', errors='coerce')
                ba = ba.dropna(subset=["trade_date", "ts_code"])
                ba = ba.drop_duplicates(["ts_code", "trade_date"], keep="last")
                # P-20260909-A: keys-only 逐列注入 (同 moneyflow)
                m = df[["ts_code", "trade_date"]].merge(ba, on=["ts_code", "trade_date"], how="left")
                for col in _val_cols:
                    if col in m.columns:
                        df[col] = m[col].values
                        df[f"{col}_p"] = df[col]
                del m
                print(f"  [S5] Loaded valuation(daily_basic): "
                      f"{len([c for c in _val_cols if c in df.columns])} fields")
            except Exception as e:
                print(f"  [S5] ⚠️ daily_basic load failed: {e}")

        # ── v0.10: 财务质量字段 (fina_indicator, ann_date 公告日 asof 无前视) ──
        fina_path = RAW_DIR / "fina_indicator.csv"
        if fina_path.exists():
            try:
                _fina_cols = ["roe_dt", "tr_yoy", "netprofit_yoy"]
                fi = pd.read_csv(
                    fina_path,
                    usecols=[c for c in ["ts_code", "ann_date"] + _fina_cols
                             if c in pd.read_csv(fina_path, nrows=0).columns],
                    low_memory=False)
                fi["ann_date"] = pd.to_datetime(
                    fi["ann_date"].astype(str).str.replace(r"\.0$", "", regex=True),
                    format='mixed', errors='coerce')
                fi = fi.dropna(subset=["ann_date", "ts_code"])
                fi = fi.drop_duplicates(["ts_code", "ann_date"], keep="last")
                fi = fi.sort_values(["ts_code", "ann_date"])
                df_sorted = df.sort_values(["ts_code", "trade_date"]).copy()
                fi_sorted = fi.rename(columns={"ann_date": "trade_date"})
                fi_sorted["trade_date"] = pd.to_datetime(
                    fi_sorted["trade_date"]).astype("datetime64[ns]")
                # merge_asof: 每只股票每个交易日对齐最近一次已公告的财报 (backward)
                # P-20260909-A 内存修复: keys-only asof + _k 还原 (同 balancesheet)。
                # 🔴 pandas 3.x 精度坑: 右帧 datetime 列须显式 .astype('datetime64[ns]')
                # (默认推断 <M8[us] 与左帧 ns 不匹配 → incompatible merge keys)。
                fi_sorted = fi_sorted.sort_values("trade_date")
                fi_sorted["trade_date"] = fi_sorted["trade_date"].astype("datetime64[ns]")
                _left = df_sorted[["ts_code", "trade_date"]].reset_index(drop=True)
                _left["trade_date"] = _left["trade_date"].astype("datetime64[ns]")
                _left["_k"] = np.arange(len(_left))
                _order = _left["trade_date"].values.argsort(kind="stable")
                _sorted_left = _left.iloc[_order].reset_index(drop=True)
                _merged = pd.merge_asof(
                    _sorted_left, fi_sorted,
                    by="ts_code", on="trade_date", direction="backward",
                    tolerance=pd.Timedelta(days=370),
                ).set_index("_k").sort_index()
                # 对齐安全防御 (同 balancesheet)
                _aligned_ok = (
                    (df.index == df_sorted.index).all()
                    and (_merged.index == np.arange(len(df))).all()
                )
                if _aligned_ok:
                    _vals = {col: _merged[col].values for col in _fina_cols
                             if col in _merged.columns}
                else:
                    _mi = pd.Series(np.arange(len(df)), index=df_sorted.index)
                    _vals = {col: _merged[col].reindex(_mi.reindex(df.index).values).values
                             for col in _fina_cols if col in _merged.columns}
                for col in _fina_cols:
                    if col in _vals:
                        df[col] = _vals[col]
                        df[f"{col}_p"] = df[col]
                del _left, _order, _sorted_left, _merged, _vals
                print(f"  [S5] Loaded fina quality(ann_date asof): {len(_fina_cols)} fields")
            except Exception as e:
                print(f"  [S5] ⚠️ fina_indicator load failed: {e}")

        # ── v0.6.1 (2026-08-29): 龙虎榜字段 (top_list.csv → lhb_flag 等) ──
        # 修复: S5 环境缺 lhb_flag → 库内 lhb 类模板 eval 报 NameError 全军覆没
        top_path = RAW_DIR / "top_list.csv"
        if top_path.exists():
            try:
                tl = pd.read_csv(
                    top_path,
                    usecols=[c for c in ["trade_date", "ts_code", "l_buy", "l_amount",
                                         "net_amount", "net_rate"]
                             if c in pd.read_csv(top_path, nrows=0).columns],
                    low_memory=False)
                tl["trade_date"] = pd.to_datetime(
                    tl["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                tl = tl.dropna(subset=["trade_date", "ts_code"])
                # 同日同股多次上榜 → 计数为 lhb_flag, 金额类累加
                agg = tl.groupby(["ts_code", "trade_date"]).agg(
                    lhb_flag=("net_amount", "size"),
                    lhb_net_amount=("net_amount", "sum"),
                    lhb_amount=("l_amount", "sum"),
                    lhb_net_rate=("net_rate", "sum"),
                ).reset_index()
                # P-20260909-A: keys-only 逐列注入 (同 moneyflow; lhb 稀疏表 merge 后
                # 大量 NaN → 注入后 fillna, 避免整帧 merge 的 consolidate 大块)
                m = df[["ts_code", "trade_date"]].merge(agg, on=["ts_code", "trade_date"], how="left")
                for col in ["lhb_flag", "lhb_net_amount", "lhb_amount", "lhb_net_rate"]:
                    if col in m.columns:
                        df[col] = m[col].values
                        df[col] = df[col].fillna(0)
                        df[f"{col}_p"] = df[col]
                del m
                print(f"  [S5] Loaded top_list: 4 fields (lhb_flag, lhb_net_amount, "
                      f"lhb_amount, lhb_net_rate)")
            except Exception as e:
                print(f"  [S5] ⚠️ top_list load failed: {e}")
        else:
            print(f"  [S5] ⚠️ top_list.csv not found — lhb 类模板将失败")

        # ── 🔴 P-20260918-001 (2026-09-18): 龙虎榜**席位**三路通道 (top_inst.csv) ──
        # 死接线修复: S5 per-stock eval 上下文此前**只有** top_list 的 4 个汇总字段
        # (lhb_flag/net_amount/amount/net_rate), 完全没有席位级三路通道
        # (lhb_inst_* / lhb_north_* / lhb_yz_*) ⇒ 库内 novel_234/235/236
        # (P-20260917-022 的三路资金因子) 在 S5 层 eval 报 NameError 全军覆没。
        # 口径与 data_loader_ext.load_top_list_wide / factor_ic_computer 严格同式。
        # env FUXI_LHB_TRI_CHANNEL = on(默认, 修复后) | off(修正前 = 仅 top_list 4 字段)
        if _lhb_tri_on():
            ti_path = RAW_DIR / "top_inst.csv"
            if ti_path.exists():
                try:
                    ti = pd.read_csv(ti_path, low_memory=False,
                                     dtype={"ts_code": str})
                    ti["trade_date"] = pd.to_datetime(
                        ti["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                    ti = ti.dropna(subset=["trade_date", "ts_code"])
                    for c in ["buy", "sell", "net_buy"]:
                        ti[c] = pd.to_numeric(ti[c], errors="coerce")
                    _ex = ti["exalter"].astype(str)
                    _inst = ti[_ex.str.contains("机构专用", na=False)]
                    _north = ti[_ex.str.contains(
                        "深股通专用|沪股通专用", na=False, regex=True)]
                    _yz = ti[~_ex.str.contains(
                        "机构专用|深股通专用|沪股通专用", na=False, regex=True)]
                    _yz = _yz[_yz["exalter"].astype(str).str.contains(
                        "营业部", na=False)]
                    _parts = []
                    for _k, _sub in (("inst", _inst), ("north", _north), ("yz", _yz)):
                        if len(_sub) == 0:
                            continue
                        _a = _sub.groupby(["ts_code", "trade_date"], as_index=False).agg(**{
                            f"lhb_{_k}_net_buy": ("net_buy", "sum"),
                            f"lhb_{_k}_buy": ("buy", "sum"),
                            f"lhb_{_k}_sell": ("sell", "sum"),
                        })
                        _parts.append(_a)
                    if _parts:
                        _agg = _parts[0]
                        for _a in _parts[1:]:
                            _agg = _agg.merge(_a, on=["ts_code", "trade_date"], how="outer")
                        _cols = [c for c in [
                            "lhb_inst_net_buy", "lhb_inst_buy", "lhb_inst_sell",
                            "lhb_north_net_buy", "lhb_north_buy", "lhb_north_sell",
                            "lhb_yz_net_buy", "lhb_yz_buy", "lhb_yz_sell"]
                            if c in _agg.columns]
                        m2 = df[["ts_code", "trade_date"]].merge(
                            _agg, on=["ts_code", "trade_date"], how="left")
                        for col in _cols:
                            df[col] = m2[col].values
                            df[col] = df[col].fillna(0)   # 未上榜=0 (与既有口径同)
                            df[f"{col}_p"] = df[col]
                        del m2, _agg, _parts
                        print(f"  [S5] Loaded top_inst(席位三路): {len(_cols)} fields "
                              f"(inst/north/yz, 未上榜=0)")
                    del ti, _inst, _north, _yz, _ex
                except Exception as e:
                    print(f"  [S5] ⚠️ top_inst load failed: {e}")
            else:
                print(f"  [S5] ⚠️ top_inst.csv not found — 三路席位模板将失败")

        # v0.10: 慢变量按股 ffill (估值日缺失 1-2 天 / 财报公告间隔)
        _slow_cols = [c for c in ["pe_ttm", "pb", "dv_ratio",
                                  "roe_dt", "tr_yoy", "netprofit_yoy"] if c in df.columns]
        if _slow_cols:
            df = df.sort_values(["ts_code", "trade_date"])
            df[_slow_cols] = df.groupby("ts_code")[_slow_cols].ffill()

        self._price_df = df

    def _select_universe(self):
        """选择回测 universe: 按成交量排序取 top-N; None=全量池"""
        if self._price_df is None:
            self._load_data()

        # P-20260909-A: 全量池 (sample_stocks=None) — 不过滤, 全部 5000+ 只。
        # 与 S1-S4 全市场口径对齐, 五项裁决同池。
        if self.sample_stocks is None:
            self._universe = sorted(self._price_df["ts_code"].unique().tolist())
            return

        # v0.9.3: 不复制整帧, 直接 groupby 聚合取 top-N (内存减半)
        amount_avg = self._price_df.groupby("ts_code")["amount"].mean()
        top_stocks = amount_avg.nlargest(self.sample_stocks)
        self._universe = sorted(top_stocks.index.tolist())

    def evaluate_factor_formula(
        self,
        formula: str,
        factor_name: str = "unknown",
    ) -> Optional[pd.DataFrame]:
        """
        评估因子公式在价格数据上的原始信号。
        
        对 universe 中每只股票的时间序列计算因子值。
        返回 DataFrame: index=(ts_code, trade_date), columns=['factor_value']
        
        公式格式: pandas 表达式，字段名为 close_p/open_p/volume_p 等。
        v0.3.1: 自动转换 DSL 格式 → pandas infix，支持 .apply(lambda)。
        """
        if not self._loaded:
            self._load_data()
        if self._price_df is None:
            return None

        # P-20260909-A: 全量池时 _price_df 本身就是 universe (裁剪在 _load_data
        # 已跳过), 避免 7M 行 isin+copy 的整帧拷贝; Top-N 池保持原裁剪语义。
        if self.sample_stocks is None:
            df = self._price_df.copy()
        else:
            df = self._price_df[self._price_df["ts_code"].isin(self._universe)].copy()
        
        # ── 公式预处理 ─────────────────────────────────────
        clean_formula = formula.strip()

        # ── v0.7.19: 外源加载器因子 (裸标识符) 直接路由, 跳过公式层求值 ──
        # 必须在 _normalize_to_pandas 之前判断: DSL 归一化会改写裸标识符。
        _loader_df = self._evaluate_loader_formula(clean_formula, factor_name)
        if _loader_df is not None:
            return _loader_df

        # v0.7: 多行迷你程序检测 — 多行公式跳过赋值剥离, 由执行器逐行处理
        _code_lines = [ln for ln in clean_formula.split("\n")
                       if ln.strip() and not ln.strip().startswith("#")]
        _is_multiline = len(_code_lines) > 1

        # 去掉赋值 (仅单行公式; 多行的 `factor = ...` 由 _parse_multiline_formula 处理)
        if not _is_multiline:
            if "=" in clean_formula and not any(op in clean_formula for op in ["<=", ">=", "==", "!="]):
                parts = clean_formula.split("=", 1)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', parts[0].strip()):
                    clean_formula = parts[1].strip()

        # v0.3.1: DSL → pandas infix 自动转换 (多行公式起点非 DSL, 原样返回)
        clean_formula = self._normalize_to_pandas(clean_formula)
        
        if not clean_formula:
            return None

        if not _is_multiline:
            clean_formula = self._normalize_rank_axis(clean_formula)

        # ── v0.7.19: 外源加载器因子 (裸标识符) 直接路由, 跳过公式层求值 ──
        # 必须在 _normalize_to_pandas 之前判断: DSL 归一化会把裸标识符改写。
        _loader_df = self._evaluate_loader_formula(clean_formula, factor_name)
        if _loader_df is not None:
            return _loader_df

        # v0.3.1: 预验证 — 检查公式中的字段引用是否在已知字段中
        unknown_fields = self._check_unknown_fields(clean_formula)
        if unknown_fields:
            print(f"    [S5] {factor_name}: WARNING unknown fields {unknown_fields}, will still try eval")
            # 不直接拒绝，给 eval 一个机会（某些字段可能是通过 pandas 方法链推导的）

        # v0.3.1: 检测需 DataFrame 上下文的公式 (mean(axis=1), .sub(..., axis=0) 等)
        needs_cross_section = self._needs_cross_section(clean_formula)

        try:
            if needs_cross_section:
                # ── 全面板模式: 构建多股票 DataFrame 上下文 ──
                return self._evaluate_cross_section(df, clean_formula, factor_name)
            # ── 单股票模式: 逐股 eval ──
            _res = self._evaluate_per_stock(df, clean_formula, factor_name)
            # 2026-09-18 (P-20260918-022) 安全网: 路由判据无论多宽都可能漏。
            # 逐股路径**无结果**且公式含 axis 记号 ⇒ 升级到截面路径重试一次, 并留痕
            # (三态: 未走 / 升级后通 / 升级后仍空), 杜绝静默丢候选。
            # 只在 Series 路径已失败时触发 ⇒ 对当前能通过的公式**零行为变化**。
            if (_res is None or len(_res) == 0) and 'axis' in clean_formula:
                print(f"    [S5] {factor_name}: 逐股路径无结果且公式含 axis 记号 "
                      f"→ 升级到截面路径重试 (留痕)")
                _res2 = self._evaluate_cross_section(df, clean_formula, factor_name)
                if _res2 is not None and len(_res2):
                    print(f"    [S5] {factor_name}: 升级成功 ({len(_res2)} 行) — "
                          f"说明路由判据曾漏判, 请补正则")
                else:
                    print(f"    [S5] {factor_name}: 升级后仍无结果 (截面路径亦空/报错)")
                return _res2
            return _res
        except Exception as e:
            print(f"    [S5] Error evaluating {factor_name}: {e}")
            return None

    def _evaluate_loader_formula(
        self, formula: str, factor_name: str = "unknown"
    ) -> Optional[pd.DataFrame]:
        """v0.7.19: 裸标识符公式 → data_loader_ext 专用加载器宽表 → 日频长表。

        返回 None 表示「不是加载器公式」(交给常规公式层); 抛异常或空表时也会
        打印诊断并返回 None, 由调用方的常规路径继续处理 (不静默吞掉)。

        契约: 返回 [ts_code, trade_date, factor_value] 长表, 已裁到 universe。
        """
        if _loader_formula_switch() == "off":
            return None
        name = (formula or "").strip()
        if name not in LOADER_FORMULA_REGISTRY:
            return None
        fn_name, slot = LOADER_FORMULA_REGISTRY[name]
        try:
            import data_loader_ext as _dle
        except Exception:  # noqa: BLE001 — 独立调用时 scripts/ 可能不在 sys.path
            try:
                _sd = str(Path(__file__).resolve().parent.parent.parent / "scripts")
                if _sd not in sys.path:
                    sys.path.insert(0, _sd)
                import data_loader_ext as _dle  # noqa: F811
            except Exception as e:  # noqa: BLE001
                print(f"    [S5-load] {factor_name}: data_loader_ext 不可用 ({e})")
                return None
        try:
            fn = getattr(_dle, fn_name)
        except AttributeError as e:
            print(f"    [S5-load] {factor_name}: 加载器 {fn_name} 不存在 ({e})")
            return None
        try:
            index = pd.DatetimeIndex(sorted(self._price_df["trade_date"].unique()))
            columns = sorted(self._price_df["ts_code"].unique())
            _out = fn(index, columns)
            if isinstance(slot, str):
                # v0.7.20: dict 型加载器 → 按字段名取 (缺失键不算错误, 打印后可归因)
                wide = _out.get(slot) if isinstance(_out, dict) else None
                if wide is None:
                    _have = len(_out) if isinstance(_out, dict) else type(_out).__name__
                    print(f"    [S5-load] {factor_name}: 加载器 {fn_name} 无字段 "
                          f"'{slot}' (可用: {_have})")
                    return None
            else:
                wide = _out[slot]
        except Exception as e:  # noqa: BLE001
            print(f"    [S5-load] {factor_name}: 加载器 {fn_name} 失败 "
                  f"({type(e).__name__}: {str(e)[:80]})")
            return None
        if wide is None or wide.empty:
            print(f"    [S5-load] {factor_name}: 加载器 {fn_name} 返回空表")
            return None

        long = (wide.stack().rename("factor_value").reset_index())
        long.columns = ["trade_date", "ts_code", "factor_value"]
        long["trade_date"] = pd.to_datetime(long["trade_date"])
        long = long[np.isfinite(long["factor_value"].to_numpy(dtype="float64"))]
        if self.sample_stocks is not None:
            long = long[long["ts_code"].isin(self._universe)]
        print(f"    [S5-load] {factor_name}: 加载器通道 {fn_name}[{slot}] → "
              f"{len(long)} 行 / {long['ts_code'].nunique()} 只 / "
              f"覆盖率 {long['ts_code'].nunique()/max(len(columns),1)*100:.1f}%")
        return long[["ts_code", "trade_date", "factor_value"]].reset_index(drop=True)

    def _normalize_to_pandas(self, formula: str) -> Optional[str]:
        """v0.3.1: 将 DSL 格式公式转为 pandas infix，如果已是则原样返回"""
        # 快速检测：DSL 格式以 func_name( 开头
        dsl_pattern = re.match(
            r'^(sub|add|mul|div|neg|abs|sqrt|log|sign|ma|ts_mean|ts_std|'
            r'ts_min|ts_max|ts_sum|ts_delta|ts_delay|delta|ema|roc|rank|'
            r'zscore|cs_zscore|demean|normalize|ts_corr|ts_cov|ts_rank|'
            r'ts_skewness|ts_kurtosis|ts_regression|ts_decay_linear)\(', 
            formula
        )
        if not dsl_pattern:
            return formula  # 已经是 pandas infix 或其他格式
        
        # 调用因子表达式解析器的转换功能
        try:
            # 注入 factor_expression_tree 模块路径
            import sys
            from pathlib import Path
            _pkg = Path(__file__).parent
            if str(_pkg) not in sys.path:
                sys.path.insert(0, str(_pkg))
            from factor_expression_tree import dsl_to_pandas_infix
        except ImportError:
            return formula  # 无法导入，用原公式
        
        converted = dsl_to_pandas_infix(formula)
        if converted:
            return converted
        return formula

    def _check_unknown_fields(self, formula: str) -> set:
        """v0.3.1: 检查公式中是否有不在 S5 数据中的字段引用"""
        import re
        
        # S5 已知的所有字段名
        known = {
            "close_p", "open_p", "high_p", "low_p", "volume_p", "amount_p",
            "close", "open", "high", "low", "volume", "amount",
            "overnight", "overnight_p", "amplitude", "amplitude_p",
            "returns", "returns_p", "turnover", "turnover_p",
            "hl_ratio", "hl_ratio_p",
            # P-018/P-008/P-009: moneyflow + balancesheet 字段
            "buy_lg_vol", "sell_lg_vol", "buy_sm_vol", "sell_sm_vol",
            "buy_md_vol", "sell_md_vol", "buy_elg_vol", "sell_elg_vol",
            # v0.6.2 (2026-08-29): moneyflow 净流入字段 (对齐 IC Computer 10 字段)
            "net_mf_vol", "net_mf_amount",
            "net_mf_vol_p", "net_mf_amount_p",
            # v0.6.2 (2026-08-29): 两融字段 (margin_detail.csv, 对齐 IC Computer)
            # P-20260911-margin (2026-09-11): 扩展至 8 字段 (新增 rzche/rqmcl/rzrqye)
            *_MARGIN_COLS,
            *[f"{c}_p" for c in _MARGIN_COLS],
            "total_assets", "intan_assets", "goodwill",
            "total_assets_p", "intan_assets_p", "goodwill_p",
            # v0.7: 基本面季度数据 (fina_batch)
            "roe", "eps", "eps_qoq",
            # v0.10 (2026-08-25 红利审计): 估值 + 财务质量字段
            "pe_ttm", "pb", "dv_ratio", "roe_dt", "tr_yoy", "netprofit_yoy",
            "pe_ttm_p", "pb_p", "dv_ratio_p", "roe_dt_p", "tr_yoy_p", "netprofit_yoy_p",
            # v0.6.1 (2026-08-29): 龙虎榜字段
            "lhb_flag", "lhb_net_amount", "lhb_amount", "lhb_net_rate",
            "lhb_flag_p", "lhb_net_amount_p", "lhb_amount_p", "lhb_net_rate_p",
        }
        
        # 内置/关键词（非字段引用）
        keywords = {
            "rolling", "shift", "pct_change", "mean", "std", "skew", "kurt",
            "corr", "cov", "rank", "apply", "lambda", "abs", "sqrt", "log",
            "np", "pd", "range", "len", "sum", "min", "max", "int", "float",
            "True", "False", "None", "astype", "fillna", "clip", "where",
            "exp", "sign", "log1p", "round", "sub", "div", "mul", "add", "neg",
            "cumsum", "cumprod", "diff", "corrcoef", "x", "raw", "ewm", "span",
            "polyfit", "arange", "quantile", "replace", "nan", "inf",
            "min_periods", "adjust", "axis",
            # clip 关键字参数 (P-20260910-008: 周频种子 volume.clip(lower=1) 误报 unknown)
            "lower", "upper", "inplace", "limit", "subset",
            "list", "dict", "tuple", "str", "bool", "sorted", "zip", "isinstance",
            # v0.5 伏羲: 常见简写和别名
            "pct", "pct_chg", "ret", "vwap", "mv",
            "maximum", "minimum",
        }
        
        # 提取所有标识符
        cleaned = re.sub(r"'[^']*'", '', formula)
        cleaned = re.sub(r'"[^"]*"', '', cleaned)
        all_words = set(re.findall(r'[a-zA-Z_]\w*', cleaned))
        
        refs = all_words - keywords
        refs = {w for w in refs if not w.replace('_', '').isdigit()}
        refs = {w for w in refs if len(w) > 1}
        
        return refs - known

    def _check_formula_sanity(
        self, formula: str, factor_name: str, skip_simplify: bool = False
    ) -> Optional[str]:
        """
        v0.4: 检查公式的金融合理性，拦截垃圾因子。
        
        Returns:
          None if formula passes sanity check
          str with rejection reason if formula is garbage
        """
        import re
        
        # ── 规则1: 冗余 np.abs —— 价格字段恒正, abs 无意义 ──
        price_fields = ['close_p', 'open_p', 'high_p', 'low_p', 'volume_p', 'amount_p',
                        'close', 'open', 'high', 'low', 'volume', 'amount']
        for pf in price_fields:
            if f'np.abs({pf})' in formula or f'abs({pf})' in formula:
                return f"冗余abs: {pf}恒正, np.abs无意义"
            if re.search(rf'np\.abs\(\s*{re.escape(pf)}\s*\)', formula):
                return f"冗余abs: {pf}恒正, np.abs无意义"
        
        # ── 规则2: 价格-百分比量纲混合 ──
        # 检测 price_field - something_that_returns_percentage
        price_pattern = r'(close_p|open_p|high_p|low_p|close|open|high|low)\s*[-+]\s*\S*\.pct_change'
        if re.search(price_pattern, formula):
            # Check if the pct_change result is subtracted from a price level
            # This is almost always wrong (price_level - return_percentage)
            return f"量纲混合: 价格水平字段与pct_change(百分比)直接加减, 单位不一致"
        
        # ── 规则3: 价格水平 - 常数 → 本质是价格代理 ──
        # e.g., close_p - 1, high_p + 0.001
        # 但排除: (field1 - price) + epsilon 的除零保护形式
        stripped = formula.strip()
        _sanity_mode = os.environ.get('FUXI_S5_SANITY_AST', 'enforce').strip().lower()
        if _sanity_mode not in ('enforce', 'legacy'):
            raise ValueError('FUXI_S5_SANITY_AST must be enforce or legacy')
        if _sanity_mode == 'enforce':
            import ast
            try:
                root = ast.parse(stripped, mode='eval').body
            except SyntaxError:
                root = None
            # Only a root field +/- literal is a field-level proxy. A nested
            # denominator epsilon is not evidence about the whole factor.
            if (isinstance(root, ast.BinOp) and isinstance(root.op, (ast.Add, ast.Sub))
                    and isinstance(root.left, ast.Name) and root.left.id in price_fields
                    and isinstance(root.right, ast.Constant)
                    and isinstance(root.right.value, (int, float))
                    and 0.01 < abs(root.right.value) < 100):
                return f'价格代理: {root.left.id} 与常数平移, 非alpha因子'
        for pf in price_fields:
            if _sanity_mode == 'enforce':
                break
            # 排除 / price_field - N 的收益率模式
            if re.search(rf'/\s*{re.escape(pf)}\s*-\s*\d+', stripped):
                continue
            # 排除 price_field 后紧跟 ) 的复合表达式 (如 amount_p - open_p) + 0.001)
            if re.search(rf'{re.escape(pf)}\s*\)\s*[-+]', stripped):
                continue
            # 找 price_field -/+ 数字的模式（price_field 不能是复合表达式的一部分）
            m = re.search(
                rf'(?:^|[^-+*/]){re.escape(pf)}\s*[-+]\s*(\d+\.?\d*)\s*[^.]',
                stripped
            )
            if m:
                const_val = float(m.group(1))
                # >0.01 的常数才算垃圾（<=0.01 可能是 epsilon）
                if const_val > 0.01 and const_val < 100:
                    return f"价格代理: {pf} - {const_val} ≈ 价格水平本身, 非alpha因子"
        
        # ── 规则4: 价格水平 + 微小百分比调整 → 本质是价格代理 ──
        # e.g., high_p - close_p.pct_change(20).shift(20) → 99.7% high_p
        for pf in price_fields:
            # Check if formula is price_field +/- (something with pct_change and shift)
            m = re.search(
                rf'{re.escape(pf)}\s*[-+]\s*\(?\s*\S*\.pct_change\(\d+\).*',
                stripped
            )
            if m:
                return f"价格代理: {pf}与微小百分比调整(shift)混合, 99%+由价格水平主导"
        
        # ── 规则5: 简单价格滚动均值无标准化 → 价格趋势代理, 非横截面alpha ──
        # e.g., close_p.rolling(N).mean() without any normalization
        for pf in price_fields:
            # 去空格和外围括号
            core = re.sub(r'\s+', '', stripped)
            core = re.sub(r'^\((.*)\)$', r'\1', core)
            # 匹配: price_field.rolling(N).mean() 或 -price_field.rolling(N).mean()
            if re.match(rf'^-?{re.escape(pf)}\.rolling\(\d+\)\.mean\(\)$', core):
                return f"价格趋势代理: {pf}.rolling(N).mean() 无标准化, 非横截面alpha"
        
        # ── 规则6: 极简公式 (少于2次pandas操作出现) ──
        # v0.6.1 (2026-08-29): seed 重检豁免 — 种子原式 (如 open/close.shift(1)-1)
        # 天然简单且已被 JQ 验证过, 重检目的是确认有效性而非查新颖性
        if skip_simplify:
            return None
        pandas_ops = [
            'rolling', 'pct_change', 'shift', '.mean(', '.std(', '.skew(',
            '.corr(', '.cov(', '.rank(', '.diff(', '.sub(', '.div(', '.mul(', '.add(',
            '.min(', '.max(', '.sum(', '.prod(', '.quantile(',
            'rolling_min', 'rolling_max', 'rolling_std', 'rolling_sum',
            '.abs(', '.apply(', '.fillna(', '.clip(', '.replace(',
            '.cumsum(', '.cumprod(', '.ewm(',
        ]
        op_count = sum(formula.count(op) for op in pandas_ops)
        if op_count < 2:
            return f"过度简化: 仅{op_count}次pandas操作, 可能为原始价格/量的线性变换"
        
        return None  # passes sanity

    @staticmethod
    def _normalize_rank_axis(formula: str) -> str:
        mode = os.environ.get('FUXI_S5_RANK_AXIS', 'enforce').strip().lower()
        if mode == 'legacy':
            return formula
        if mode != 'enforce':
            raise ValueError('FUXI_S5_RANK_AXIS must be enforce or legacy')
        try:
            tree = ast.parse(formula, mode='eval')
        except SyntaxError:
            return formula
        changed = False

        class RankAxis(ast.NodeTransformer):
            def visit_Call(self, node):
                nonlocal changed
                node = self.generic_visit(node)
                if not isinstance(node.func, ast.Attribute) or node.func.attr != 'rank':
                    return node
                receiver = node.func.value
                if (isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute)
                        and receiver.func.attr in ('rolling', 'expanding', 'groupby')):
                    return node
                axes = [kw.value for kw in node.keywords if kw.arg == 'axis']
                if node.args:
                    axes = [node.args[0]]
                if axes:
                    axis = axes[0]
                    if not isinstance(axis, ast.Constant) or axis.value not in (1, 'columns'):
                        raise ValueError('plain rank over the complete time axis is non-causal; use rolling rank')
                else:
                    node.keywords.append(ast.keyword(arg='axis', value=ast.Constant(1)))
                    changed = True
                return node

        tree = RankAxis().visit(tree)
        return ast.unparse(ast.fix_missing_locations(tree)) if changed else formula

    def _needs_cross_section(self, formula: str) -> bool:
        """检测公式是否需要横截面 DataFrame 上下文"""
        cs_markers = [
            'mean(axis=1)', 'mean(axis = 1)', 
            'std(axis=1)', 'std(axis = 1)',
            '.sub(', '.mul(', '.add(', '.div(',
            'axis=0', 'axis = 0',
        ]
        # 检测 axis=1 相关操作（横截面均值/排名等）
        # v0.6.2 (2026-08-29): + min/max(axis=1) (Forge 截面 scale 翻译产物)
        has_axis1 = any(m in formula for m in ['mean(axis=1)', 'mean(axis = 1)',
                                               'std(axis=1)', 'std(axis = 1)',
                                               'min(axis=1)', 'min(axis = 1)',
                                               'max(axis=1)', 'max(axis = 1)',
                                               'rank(axis=1', 'rank(axis = 1',
                                               'rank(pct=True, axis=1'])
        # 2026-09-18 (P-20260918-022): 字面白名单只覆盖"恰好这么写"的形式;
        # 写法变体 (逗号后无空格 / 插入 ascending= / 参数顺序对调) 会漏判 ⇒
        # 漏判后落进逐股 Series 上下文 ⇒ Series.rank(axis=1) ValueError ⇒ 静默丢。
        # 实测: novel_217~226 改 `rank(pct=True,axis=1)` 后路由判定 True→False。
        # 此兜底**只加不减**, 不改变当前能通过公式的路由。
        if not has_axis1:
            has_axis1 = bool(_AXIS1_CALL_RE.search(formula))
        # 检测 sub(..., axis=0) 模式
        has_axis0 = any(m in formula for m in ['axis=0', 'axis = 0'])
        return has_axis1 or has_axis0

    def _evaluate_cross_section(
        self, df: pd.DataFrame, formula: str, factor_name: str
    ) -> Optional[pd.DataFrame]:
        """
        v0.3.1: 横截面评估 — 构建完整的多股票 DataFrame 上下文。
        用于处理 mean(axis=1) / .sub(axis=0) 等跨股票操作。
        """
        # 构建 pivot 表: index=trade_date, columns=ts_code
        fields = ['open_p', 'high_p', 'low_p', 'close_p', 'volume_p', 'amount_p',
                  'overnight_p', 'amplitude_p', 'returns_p', 'turnover_p']
        # P-018: 动态包含 moneyflow/balancesheet 额外字段
        for extra_col in ['buy_lg_vol', 'sell_lg_vol', 'buy_sm_vol', 'sell_sm_vol',
                          'buy_md_vol', 'sell_md_vol', 'buy_elg_vol', 'sell_elg_vol',
                          'net_mf_vol', 'net_mf_amount',
                          # v0.6.2 (2026-08-29): 两融字段 (P-20260911-margin: 扩展至 8 字段)
                          *_MARGIN_COLS,
                          'total_assets', 'intan_assets', 'goodwill',
                          # v0.10: 估值 + 财务质量字段
                          'pe_ttm', 'pb', 'dv_ratio', 'roe_dt', 'tr_yoy', 'netprofit_yoy',
                          # v0.6.1: 龙虎榜字段
                          'lhb_flag', 'lhb_net_amount', 'lhb_amount', 'lhb_net_rate']:
            if extra_col in df.columns:
                fields.append(extra_col)
        # ── 2026-09-18 (P-20260918-022): 公式驱动字段补全 ──────────────
        # 上面是硬编码清单, 与求值侧字段集合会漂移: 公式引用了清单外的列时
        # cross-section 路径 NameError → **静默丢候选**。
        # 实锤: novel_234~240 (三路龙虎榜席位因子) 因 lhb_{inst,north,yz}_* 未进
        # 清单, 在截面路径全军覆没 (逐股路径已接线, 二者不同步)。
        # 修法: 从公式抽取标识符 ∩ df.columns 补入 fields。成本有界 (随公式长度),
        # 不采用"pivot 全部列"以避免本项目 2.48GiB 级内存前科复现。
        _KEY_COLS = ('trade_date', 'ts_code')   # 键列不参与 pivot (非数值语义)
        for _tok in sorted(set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula))):
            if (_tok in df.columns and _tok not in fields
                    and _tok not in _KEY_COLS):
                fields.append(_tok)
        available = [f for f in fields if f in df.columns]
        
        # v0.3.1: 去重（避免 pivot 时 "Index contains duplicate entries" 错误）
        df_dedup = df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='last')
        
        pivots = {}
        for field in available:
            piv = df_dedup.pivot(index='trade_date', columns='ts_code', values=field)
            pivots[field] = piv
        
        # 移除 trade_date 索引 → 纯 DataFrame
        # v0.3.1: Safe builtins for lambda closures
        safe_builtins = {
            'range': range, 'len': len, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'str': str, 'bool': bool,
            'True': True, 'False': False, 'None': None,
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sum': sum, 'zip': zip, 'sorted': sorted, 'reversed': reversed,
            'print': print, 'isinstance': isinstance,
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            # v0.7: 多行公式体含 import 语句, 需 __import__
            '__import__': __import__,
        }
        
        # v0.7: 基本面 roe/eps 面板 (质量类因子: roe / eps_qoq)
        fund_globals = {}
        try:
            if pivots:
                _ref = pivots["close_p"] if "close_p" in pivots else next(iter(pivots.values()))
                fund_pivots = self._load_fundamental_pivots(
                    target_index=_ref.index, target_columns=list(_ref.columns))
                if fund_pivots.get("eps") is not None:
                    fund_pivots["eps_qoq"] = fund_pivots["eps"].diff(1)
                fund_globals = fund_pivots
        except Exception as e:
            print(f"    [S5] fundamental pivots unavailable: {e}")
        
        full_globals = {
            "__builtins__": safe_builtins,
            **pivots,
            **{f.replace('_p', ''): pivots[f] for f in available if f.endswith('_p')},
            **fund_globals,
            "np": np, "pd": pd,
            "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
            "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            "where": np.where, "clip": np.clip,
            "range": range, "len": len, "int": int, "float": float,
            "list": list, "dict": dict,
        }
        
        try:
            # v0.7: 多行公式 — 赋值体整块 exec (保留 for/if 缩进块), 末行作为结果表达式
            body_lines, final_expr = _parse_multiline_formula(formula)
            if body_lines:
                exec("\n".join(body_lines), full_globals)
                if final_expr is None:
                    result = full_globals.get("factor")
                    if result is None:
                        result = full_globals.get("result")
                    if result is None:
                        raise ValueError("多行公式未定义 factor/result 变量")
                else:
                    result = eval(final_expr, full_globals, {})
            else:
                result = eval(formula, full_globals, {})
            
            if isinstance(result, pd.DataFrame):
                # 将宽表转回长表
                result_long = result.stack().reset_index()
                result_long.columns = ['trade_date', 'ts_code', 'factor_value']
                _n_raw = len(result_long)
                result_long = result_long.dropna(subset=['factor_value'])
                result_long = result_long[np.isfinite(result_long['factor_value'])]
                # 2026-09-18 (P-20260918-022): 原实现在此**完全静默** —
                # eval 成功但整表 NaN 时直接返回空表, 调用方无从区分
                # "公式写错" vs "真的没信号"。补明确诊断 (留痕), 不改变返回值。
                if len(result_long) == 0 and _n_raw > 0:
                    print(f"    [S5] {factor_name}: 截面求值产出 {_n_raw} 格但**全为 "
                          f"NaN/Inf** ⇒ 静默丢候选。常见成因: DataFrame(日期×股票) 与 "
                          f"Series(索引=日期) 直接 加减, pandas 按**列**对齐 ⇒ 全 NaN; "
                          f"应改 `.sub(Y, axis=0)` 按行对齐")
                return result_long
            elif isinstance(result, pd.Series):
                return pd.DataFrame({
                    'trade_date': df['trade_date'].unique()[:len(result)],
                    'ts_code': 'market',
                    'factor_value': result.values,
                })
            else:
                return None
        except Exception as e:
            print(f"    [S5] Cross-section eval failed for {factor_name}: {e}")
            return None

    def _load_fundamental_pivots(self, target_index, target_columns) -> Dict[str, pd.DataFrame]:
        """v0.7: 懒加载 fina_batch 季度 roe/eps, 对齐到价格面板 (index=trade_date, columns=ts_code)。

        返回 {'roe': wide_df, 'eps': wide_df} — 日频 reindex + ffill (季度数据在报告期之间不变)。
        与 data_loader_ext.load_fundamental_wide 的 Stage 2 行为保持一致。
        """
        cache = getattr(self, "_fina_pivots_cache", None)
        if cache is None:
            import glob as _glob
            roe_p = eps_p = None
            batch_files = sorted(_glob.glob(str(DATA_DIR / "tushare" / "fina_batch_*.csv")))
            if batch_files:
                frames = []
                for f in batch_files:
                    try:
                        fr = pd.read_csv(f, usecols=["ts_code", "end_date", "roe", "eps"],
                                         dtype={"ts_code": str, "end_date": str})
                        if len(fr):
                            frames.append(fr)
                    except Exception:
                        continue
                if frames:
                    fina = pd.concat(frames, ignore_index=True)
                    fina["end_date"] = pd.to_datetime(fina["end_date"], errors="coerce")
                    fina = fina.dropna(subset=["end_date"])
                    fina = fina[fina["ts_code"].isin(list(target_columns))]
                    roe_p = fina.pivot_table(values="roe", index="end_date",
                                             columns="ts_code", aggfunc="last")
                    eps_p = fina.pivot_table(values="eps", index="end_date",
                                             columns="ts_code", aggfunc="last")
            self._fina_pivots_cache = {"roe": roe_p, "eps": eps_p}

        out: Dict[str, pd.DataFrame] = {}
        for key in ("roe", "eps"):
            pv = self._fina_pivots_cache.get(key)
            if pv is None or pv.empty:
                continue
            aligned_cols = [c for c in target_columns if c in pv.columns]
            sub = pv[aligned_cols].reindex(target_index).ffill()
            out[key] = sub
        return out

    def _evaluate_per_stock(
        self, df: pd.DataFrame, formula: str, factor_name: str
    ) -> Optional[pd.DataFrame]:
        """v0.3.1: 逐股票评估（增强 eval 上下文，支持 lambda 闭包）"""
        results = []
        
        # ── v0.3.1: Safe builtins — lambda 闭包需要 builtins 中的 builtins ──
        safe_builtins = {
            'range': range, 'len': len, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'str': str, 'bool': bool,
            'True': True, 'False': False, 'None': None,
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sum': sum, 'zip': zip, 'sorted': sorted, 'reversed': reversed,
            'enumerate': enumerate, 'map': map, 'filter': filter,
            'print': print, 'isinstance': isinstance,
            'TypeError': TypeError, 'ValueError': ValueError,
            'Exception': Exception, 'ZeroDivisionError': ZeroDivisionError,
            # v0.7: 多行公式体含 import 语句, 需 __import__
            '__import__': __import__,
        }
        
        # ── 核心上下文（放入 globals 以支持 lambda 闭包）──
        base_context = {
            "np": np, "pd": pd,
            "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
            "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            "where": np.where, "clip": np.clip,
            "range": range, "len": len, "int": int, "float": float,
            "list": list, "dict": dict, "tuple": tuple,
        }
        
        # 按股票分组计算因子值
        first_error = None
        # v0.7: 多行公式解析一次, 每只股票重复 exec 赋值体
        body_lines, final_expr = _parse_multiline_formula(formula)
        for ts_code, group in df.groupby("ts_code"):
            group = group.sort_values("trade_date")
            
            # 构建 Series 上下文
            series_context = self._build_series_context(group)
            
            # v0.3.1: 全部放入 globals（lambda 闭包才能访问 np/range/len 等）
            full_globals = {"__builtins__": safe_builtins, **base_context, **series_context}
            
            try:
                if body_lines:
                    exec("\n".join(body_lines), full_globals)
                    if final_expr is None:
                        result_series = full_globals.get("factor")
                        if result_series is None:
                            result_series = full_globals.get("result")
                        if result_series is None:
                            raise ValueError("多行公式未定义 factor/result 变量")
                    else:
                        result_series = eval(final_expr, full_globals, {})
                else:
                    result_series = eval(formula, full_globals, {})
                
                if isinstance(result_series, (pd.Series, np.ndarray)):
                    vals = result_series.values if isinstance(result_series, pd.Series) else result_series
                elif isinstance(result_series, (int, float)):
                    vals = np.full(len(group), float(result_series))
                else:
                    continue
                
                results.append(pd.DataFrame({
                    "ts_code": ts_code,
                    "trade_date": group["trade_date"].values,
                    "factor_value": vals.ravel() if vals.ndim > 1 else vals,
                }))
            except Exception as e:
                if first_error is None:
                    first_error = (ts_code, type(e).__name__, str(e)[:80])
                continue

        if not results:
            if first_error:
                ts, err_type, err_msg = first_error
                print(f"    [S5] {factor_name}: All stocks failed. First error [{ts}]: {err_type}: {err_msg}")
            return None

        factor_df = pd.concat(results, ignore_index=True)
        factor_df = factor_df.dropna(subset=["factor_value"])
        factor_df = factor_df[np.isfinite(factor_df["factor_value"])]
        return factor_df

    def _build_series_context(self, group: pd.DataFrame) -> Dict:
        """构建单股票的 eval 上下文（所有字段转为 pandas Series）"""
        context = {}
        field_cols = [
            "close_p", "open_p", "high_p", "low_p", "volume_p", "amount_p",
            "overnight", "overnight_p", "amplitude", "amplitude_p",
            "returns", "returns_p", "turnover", "turnover_p",
            "hl_ratio", "hl_ratio_p",
            # v0.6 P-008: moneyflow fields
            "buy_lg_vol", "buy_lg_vol_p", "sell_lg_vol", "sell_lg_vol_p",
            "buy_sm_vol", "buy_sm_vol_p", "sell_sm_vol", "sell_sm_vol_p",
            "buy_md_vol", "buy_md_vol_p", "sell_md_vol", "sell_md_vol_p",
            "buy_elg_vol", "buy_elg_vol_p", "sell_elg_vol", "sell_elg_vol_p",
            # v0.6.2 (2026-08-29): moneyflow 净流入 + 两融字段
            # (缺此 NameError: net_mf_amount/rzye 未定义 → margin_mf 类 seed 重检失败)
            "net_mf_vol", "net_mf_vol_p", "net_mf_amount", "net_mf_amount_p",
            "rzye", "rzye_p", "rqye", "rqye_p", "rzmre", "rzmre_p",
            "rqyl", "rqyl_p", "rqchl", "rqchl_p",
            # v0.6 P-009: balancesheet fields
            "intan_assets", "intan_assets_p", "goodwill", "goodwill_p",
            "total_assets", "total_assets_p",
            # v0.10 (2026-08-25 红利审计): 估值 + 财务质量字段
            "pe_ttm", "pe_ttm_p", "pb", "pb_p", "dv_ratio", "dv_ratio_p",
            "roe_dt", "roe_dt_p", "tr_yoy", "tr_yoy_p",
            "netprofit_yoy", "netprofit_yoy_p",
            # v0.6.1 (2026-08-29): 龙虎榜字段 (per-stock eval 上下文,
            # 缺此 NameError: lhb_flag 未定义 → lhb 模板全军覆没)
            "lhb_flag", "lhb_flag_p", "lhb_net_amount", "lhb_net_amount_p",
            "lhb_amount", "lhb_amount_p", "lhb_net_rate", "lhb_net_rate_p",
        ]
        # 🔴 P-20260918-001: 龙虎榜席位三路通道 (inst/north/yz 各 3 + _p)
        # 缺此则库内 novel_234/235/236 三路资金因子在 S5 层 NameError 全军覆没。
        if _lhb_tri_on():
            field_cols += [_c for _c in _LHB_TRI_COLS] + [f"{_c}_p" for _c in _LHB_TRI_COLS]
        for col in field_cols:
            if col in group.columns:
                context[col] = pd.Series(group[col].values, index=group.index)
        
        # 无 _p 后缀版本
        suffix_map = {"close": "close_p", "open": "open_p", "high": "high_p",
                      "low": "low_p", "volume": "volume_p", "amount": "amount_p"}
        for short, full in suffix_map.items():
            if full in context:
                context[short] = context[full]
        
        return context

    # ── v0.7.23 P0-a: 涨跌停面板 (惰性, 宽表) ──────────────────────
    def _get_limit_up_wide(self) -> Optional[pd.DataFrame]:
        """``stk_limit.csv`` → **up_limit 宽表** (index=日期, columns=ts_code)。

        只在 ``FUXI_S5_TRADABLE_GATE=on`` 时被调用 (默认路径零内存/零耗时)。

        内存设计: stk_limit 有 8.78M 行, 若以长表 + object 型 ts_code 常驻
        ≈ 700MB 峰值 (本项目有过 2.48GiB `_ArrayMemoryError` 前科)。故:
          1) ``dtype={'ts_code':'category'}`` → 代码列只存 2 字节编码 (17MB);
          2) 用 numpy 花式索引直接落成 **float32 宽表** (1,350×5,763 ≈ 31MB),
             避开 ``unstack`` 对 category 的未观测类别告警;
          3) 长表立即释放。
        实测覆盖至 2026-08-03 ⇒ 之后无行, 调用方回落 ``_board_limit_vector``。
        """
        if self._limit_panel is not None:
            return self._limit_panel if len(self._limit_panel) else None
        _p = RAW_DIR / "stk_limit.csv"
        if not _p.exists():
            print(f"    [S5-tradable] stk_limit.csv 缺失 ({_p}) → 全程兜底判定")
            self._limit_panel = pd.DataFrame()
            return None
        try:
            # 🔴 trade_date 是**混合格式**列 (实测既有 '20210104' 又有 '2026-05-28'),
            # 指定 int64 会整表读失败 (ValueError: invalid literal for int()
            # with base 10: '2026-05-28') → 被迫全程走兜底, 精确法形同未上线。
            # 故按 str 读入, 交给 _parse_yyyymmdd 两段式解析。
            lim = pd.read_csv(_p, usecols=["ts_code", "trade_date", "up_limit"],
                              dtype={"ts_code": "category", "trade_date": "str",
                                     "up_limit": "float32"}, low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"    [S5-tradable] stk_limit 读取失败 "
                  f"({type(e).__name__}: {str(e)[:80]}) → 兜底判定")
            self._limit_panel = pd.DataFrame()
            return None
        # 🔴 纯整数 YYYYMMDD 必须显式 %Y%m%d (format='mixed' 会当 epoch 误解析)
        _dt = _parse_yyyymmdd(lim["trade_date"])
        lim = lim.assign(_dt=_dt).dropna(subset=["_dt"])
        lim = lim[lim["up_limit"].notna()]
        if self._universe:
            _uset = set(self._universe)
            lim = lim[lim["ts_code"].isin(_uset)]
        if self._price_df is not None and len(self._price_df):
            _d0 = self._price_df["trade_date"].min()
            _d1 = self._price_df["trade_date"].max()
            lim = lim[(lim["_dt"] >= _d0) & (lim["_dt"] <= _d1)]
        lim = lim.drop_duplicates(subset=["_dt", "ts_code"], keep="last")
        if len(lim) == 0:
            self._limit_panel = pd.DataFrame()
            return None

        _dates = np.sort(lim["_dt"].unique())
        _codes = list(lim["ts_code"].cat.categories)
        _row = pd.Index(_dates).get_indexer(lim["_dt"].to_numpy())
        _col = lim["ts_code"].cat.codes.to_numpy()
        _ok = (_row >= 0) & (_col >= 0)
        _mat = np.full((len(_dates), len(_codes)), np.nan, dtype="float32")
        _mat[_row[_ok], _col[_ok]] = lim["up_limit"].to_numpy()[_ok]
        wide = pd.DataFrame(_mat, index=pd.DatetimeIndex(_dates), columns=_codes)
        del lim, _mat
        import gc as _gc
        _gc.collect()
        print(f"    [S5-tradable] stk_limit → up_limit 宽表 "
              f"{wide.shape[0]} 日 × {wide.shape[1]} 只 "
              f"(有效 {int(np.isfinite(wide.to_numpy()).sum()):,} 格, "
              f"至 {wide.index.max().date()}) → 此后走兜底")
        self._limit_panel = wide
        return wide

    def backtest_factor(
        self,
        factor_df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        简单回测: 周频调仓，Top-N 等权做多。

        Returns:
          portfolio_returns: Series (trade_date index) 组合日收益率
          turnover_history: Series (trade_date index) 每次调仓的换手率

        ── v0.7.23: P0/P1 六项口径修正接入点 ──────────────────────────
        全部开关**默认 off / 0 = 逐位复现修正前读数** (零回归); 各项缺陷、
        依据与实测边际影响见模块头 ``_s5_*`` 各函数与
        ``research/s5_p0_ablation.py`` (基线
        ``reports/_tmp/s5_p0_ablation.GOLDEN_prechange.json``)。

        🔴 P0-b × P1 交互 (两项**不可**简单叠加):
          - 收益定义 = ``close_p.pct_change()`` (依赖切片内前一行) ⇒ 切片必须
            含建仓日才能让每期首日拿到前一行, 故 P0-b 用 ``>=``;
          - 收益定义 = ``close_p/pre_close_p - 1`` (自洽, 不依赖前一行) ⇒ 切片
            必须**排除**建仓日, 否则会把"买入前那一刻的当日涨幅"计入 ⇒
            无论 P0-b 是否 on, 一律用 ``>``。
          两条路径各自都能且只能补回"建仓日→次一日"那一个缺失收益。
        """
        # ── M11: 账本口径分流 (FUXI_S5_LEDGER; 默认 proxy ⇒ 本函数原路径逐位不变) ──
        if False:  # M11 zero-regression: pre-change simulation
            return self._backtest_factor_fixed_shares(factor_df)

        # ── 口径快照 ──
        _adj = _s5_adj_price()
        _gate = _s5_tradable_gate()
        _first = _s5_first_day()
        _tie = _s5_tie_rule()
        _bps = _s5_cost_bps()
        _min_cov = _s5_min_coverage()
        _mark = _s5_holdings_mark()
        _mark_diag = {'missing_held_price_cells': 0, 'missing_held_price_cells_by_year': {},
                      'incomplete_marking_days': 0, 'marking_samples': [], 'carried_periods': 0}

        _switches = {"adj_price": _adj, "tradable_gate": _gate, "first_day": _first,
                     "tie_rule": _tie, "cost_bps": _bps, "min_coverage": _min_cov,
                     "holdings_mark": _mark}

        _empty = (pd.Series(dtype=float), pd.Series(dtype=float))

        def _set_diag(*, possible=0, used=0, skipped=0, samples=None,
                      blk_up=0, blk_fb=0, blk_dn=0, cost=0.0, turns=None,
                      adj_clipped=0, sparse=None):
            _rate = (used / possible) if possible else 0.0
            self._last_backtest_diag = {
                "n_periods_possible": int(possible),
                "n_periods_used": int(used),
                "n_periods_skipped": int(skipped),
                "coverage_rate": float(_rate),
                "coverage_ok": bool(_min_cov <= 0 or _rate >= _min_cov),
                "min_coverage_required": float(_min_cov),
                "skipped_samples": list(samples or []),
                "n_blocked_limit_up": int(blk_up),
                "n_blocked_fallback": int(blk_fb),
                "n_blocked_limit_down": int(blk_dn),
                "n_adj_outlier_clipped": int(adj_clipped),
                "total_cost": float(cost),
                "avg_turnover": float(np.mean(turns)) if turns else 0.0,
                "switches": _switches,
                # P-20260916-014a 稀疏结构体检 (纯披露; None = 未采集/已关闭)
                "sparse_diag": sparse,
                **_mark_diag,
                "accounting_ok": not any(_mark_diag['missing_held_price_cells_by_year'].get(str(y), 0)
                                         for y in (2025, 2026)),
                "marking_complete_full_window": _mark_diag['incomplete_marking_days'] == 0,
                "weighting_contract": "daily_equal_weight_membership_weekly; not a fixed-share NAV ledger",
            }

        if factor_df is None or len(factor_df) == 0:
            _set_diag()
            return _empty

        # 合并价格数据 (v0.7.23: 增加 amount 供并列打散键 / pre_close 供复权口径)
        _pcols = ["ts_code", "trade_date", "close", "close_p", "amount"]
        for _c in ("pre_close", "pre_close_p"):
            if _c in self._price_df.columns:
                _pcols.append(_c)
        price = self._price_df[self._price_df["ts_code"].isin(self._universe)][
            [c for c in _pcols if c in self._price_df.columns]
        ].copy()

        if _mark == 'enforce':
            for frame, label in ((price, 'price'), (factor_df, 'signal')):
                if frame.duplicated(['ts_code', 'trade_date']).any():
                    raise ValueError('duplicate ' + label + ' stock/date keys')

        merged = factor_df.merge(price, on=["ts_code", "trade_date"], how="inner")
        if len(merged) == 0:
            _set_diag()
            return _empty

        _has_pre = (_adj != "off") and ("pre_close_p" in merged.columns)

        # P0-a: up_limit 宽表 (惰性加载; 默认 off 时完全不加载)
        _ul_wide = self._get_limit_up_wide() if _gate else None

        # 按周分组，取每周最后一个交易日
        calendar_source = price if _mark == 'enforce' else merged
        calendar_source = calendar_source[['trade_date']].drop_duplicates().copy()
        if _mark == 'enforce':
            calendar_source = calendar_source[calendar_source.trade_date >= factor_df.trade_date.min()]
        calendar_source['week'] = calendar_source.trade_date.dt.to_period('W-SUN')
        
        # 每周最后一天
        week_last = calendar_source.groupby("week")["trade_date"].max().reset_index()
        week_last.columns = ["week", "rebalance_date"]
        
        rebalance_dates = sorted(week_last["rebalance_date"].unique())
        market_dates = pd.DatetimeIndex(calendar_source.trade_date.unique()).sort_values()
        market_dates.name = 'trade_date'
        
        if len(rebalance_dates) < 2:
            _set_diag()
            return _empty

        # 逐周调仓
        portfolio_returns = []
        # P0-b × P1 交互: adj 口径下切片**必须**排除建仓日 (见 docstring)
        _lo_op = ">" if _has_pre else (">=" if _first else ">")

        n_possible = len(rebalance_dates) - 1
        n_used = 0
        n_skipped = 0
        skipped_samples = []
        n_blk_up = 0
        n_blk_fb = 0
        n_blk_dn = 0
        n_adj_clipped = 0
        total_cost = 0.0
        turns = []
        prev_codes = None
        # P-20260916-014a: 稀疏结构体检 (纯披露) 逐期收集容器
        _sparse_on = (_s5_sparse_diag() == "on")
        sparse_periods: List[Dict] = []

        for i in range(len(rebalance_dates) - 1):
            entry_date = rebalance_dates[i]
            exit_date = rebalance_dates[i + 1]
            
            # 选股: entry_date 的因子值排名
            entry_data = merged[merged["trade_date"] == entry_date].copy()
            if _mark == 'enforce':
                entry_data = entry_data[np.isfinite(pd.to_numeric(entry_data.factor_value, errors='coerce'))]
            _n_avail = len(entry_data)

            # ── P-20260916-014a 稀疏结构体检: 截面构成 + 十分位曲线 ──
            # 在**覆盖率闸门之前**收集 ⇒ 因覆盖不足被跳过的期也在样本内
            # (诊断「信号有效性」不该被覆盖问题裁剪掉样本)。
            _rec = None
            if _sparse_on:
                try:
                    _fv = pd.to_numeric(entry_data["factor_value"], errors="coerce")
                    _fwd = None
                    _source = price if _mark == 'enforce' else merged
                    _fw = _source.loc[_source["trade_date"].isin((entry_date, exit_date)),
                                     ["ts_code", "trade_date", "close_p"]]
                    if len(_fw):
                        _pv = _fw.pivot_table(index="ts_code", columns="trade_date",
                                              values="close_p", aggfunc="last")
                        if entry_date in _pv.columns and exit_date in _pv.columns:
                            _fwd = (_pv[exit_date]
                                    / _pv[entry_date].replace(0, np.nan) - 1)
                    _rec = {
                        "date": str(pd.Timestamp(entry_date).date()),
                        "n_avail": int(_n_avail),
                        "n_pos": int((_fv > 0).sum()),
                        "n_neg": int((_fv < 0).sum()),
                        "n_zero": int((_fv == 0).sum()),
                        "n_nan": int(_fv.isna().sum()),
                        "n_signal": int((_fv.notna() & (_fv != 0)).sum()),
                        "sel_zero_frac": 0.0,
                        "sel_nan_frac": 0.0,
                        "n_sel": 0,
                        "skipped": False,
                        "decile": _decile_curve_period(entry_data, _fwd),
                    }
                    sparse_periods.append(_rec)
                except Exception:  # noqa: BLE001  纯披露, 绝不影响裁决
                    _rec = None

            # ── P0-a 可成交闸门 (仅"买不进"拦截: 涨停封板) ──
            if _gate and _n_avail:
                _ul = None
                if _ul_wide is not None and entry_date in _ul_wide.index:
                    _ul = entry_data["ts_code"].map(_ul_wide.loc[entry_date])
                else:
                    _ul = pd.Series(np.nan, index=entry_data.index, dtype="float64")
                _hit_exact = _ul.notna() & (entry_data["close_p"].to_numpy()
                                            >= _ul.to_numpy() - 1e-6)
                _miss = _ul.isna()
                # 兜底: 无 limit 行的日期按**板块感知**幅度判定 (用 pre_close 口径的
                # 真实当日涨幅; 固定 9.5% 会把创业板/科创板的 +10% 误判为封板)
                if _miss.any() and "pre_close_p" in entry_data.columns:
                    _r_entry = (entry_data["close_p"]
                                / entry_data["pre_close_p"].replace(0, np.nan) - 1)
                    _hit_fb = _miss & (_r_entry >= _board_limit_vector(
                        entry_data["ts_code"]) - 0.005)
                else:
                    _hit_fb = pd.Series(False, index=entry_data.index)
                _blocked = pd.Series(_hit_exact, index=entry_data.index) | _hit_fb
                if _blocked.any():
                    n_blk_up += int(np.sum(_hit_exact))
                    n_blk_fb += int(_hit_fb.sum())
                    entry_data = entry_data[~_blocked]

            can_rebalance = len(entry_data) >= self.top_n
            if not can_rebalance:
                n_skipped += 1
                if _rec is not None:
                    _rec["skipped"] = True
                    _rec["n_after_gate"] = int(len(entry_data))
                if len(skipped_samples) < 12:
                    skipped_samples.append(
                        {"date": str(entry_date.date()), "n_available": int(_n_avail),
                         "n_after_gate": int(len(entry_data)),
                         "need": int(self.top_n)})
                if _mark == 'legacy' or prev_codes is None:
                    continue
                _mark_diag['carried_periods'] += 1
            else:
                n_used += 1

            # 按因子值排序 (假设高值做多)
            # P0-c/d: legacy=quicksort(并列序未定义) | stable=mergesort(确定但任意)
            #         | on=因子降序 + amount 降序 + ts_code 升序 (全确定性)
            if _tie == "on":
                _by = [c for c in ("factor_value", "amount", "ts_code")
                       if c in entry_data.columns]
                _asc = [False if c in ("factor_value", "amount") else True for c in _by]
                entry_data = entry_data.sort_values(_by, ascending=_asc, kind="mergesort")
            elif _tie == "stable":
                entry_data = entry_data.sort_values("factor_value", ascending=False,
                                                    kind="mergesort")
            else:
                entry_data = entry_data.sort_values("factor_value", ascending=False)
            selected = entry_data.head(self.top_n) if can_rebalance else entry_data.iloc[:0]
            selected_codes = set(selected["ts_code"].tolist()) if can_rebalance else set(prev_codes)

            if _rec is not None:
                _selv = pd.to_numeric(selected["factor_value"], errors="coerce")
                _rec["n_sel"] = int(len(_selv))
                _rec["n_after_gate"] = int(len(entry_data))
                if len(_selv):
                    _rec["sel_zero_frac"] = float((_selv == 0).mean())
                    _rec["sel_nan_frac"] = float(_selv.isna().mean())

            # ── P0-g 交易成本 (换手口径: 卖出退出腿 + 买入进入腿) ──
            _cost = 0.0
            if _bps > 0 and can_rebalance:
                if prev_codes is None:
                    _turn_notional = 1.0          # 首期全额建仓
                else:
                    _turn_notional = (len(prev_codes - selected_codes)
                                      + len(selected_codes - prev_codes)) / float(self.top_n)
                _cost = _turn_notional * _bps / 1e4
                turns.append(float(_turn_notional))
                total_cost += _cost
            prev_codes = selected_codes

            # 持仓期间收益 (P0-b × P1 交互决定下界: 见 docstring)
            marking_source = price if _mark == 'enforce' else merged
            if (_mark == 'enforce' and not _has_pre) or (_mark == 'legacy' and _lo_op == ">="):
                _mask = ((marking_source["trade_date"] >= entry_date)
                         & (marking_source["trade_date"] <= exit_date))
            else:
                _mask = ((marking_source["trade_date"] > entry_date)
                         & (marking_source["trade_date"] <= exit_date))
            period_data = marking_source[_mask & marking_source["ts_code"].isin(selected_codes)].copy()
            
            if len(period_data) == 0 and _mark == 'legacy':
                continue

            # P0-a 诊断 (卖出侧): 持有期内触及跌停的 (日,股) 计数 — 该收益**不可实现**
            # 此处用板块感知的收益口径判定 (仅诊断, 不拦截; 引擎不建模"顺延持仓")
            if _gate and "pre_close_p" in period_data.columns:
                _r_pd = (period_data["close_p"]
                         / period_data["pre_close_p"].replace(0, np.nan) - 1)
                _lim_dn = -(_board_limit_vector(period_data["ts_code"]) - 0.005)
                n_blk_dn += int((_r_pd <= _lim_dn).sum())
            
            # 计算每日组合收益 (等权)
            if _has_pre:
                # P1: 除权调整口径 (自洽, 不依赖切片内前一行)
                _r_adj = (period_data["close_p"]
                          / period_data["pre_close_p"].replace(0, np.nan) - 1)
                if _adj == "on":
                    # P1 守卫 (推荐档): 丢弃超过板块涨跌停幅度 +0.5pp 的行。
                    # 理由: 新股/恢复上市首日无涨跌幅限制, 且其 pre_close 可能是
                    # **发行价** (实测 r 达 +200400%) ⇒ 无守卫时单个标即可把等权
                    # 日均收益抬高 200400%/top_n。这类行在 pct_change 口径下本就是
                    # NaN (无前收) 而被丢弃, 守卫只是恢复这一自然保护。
                    _guard = _board_limit_vector(period_data["ts_code"]) + 0.005
                    _bad = _r_adj.abs() > _guard
                    n_adj_clipped += int(_bad.sum())
                    _r_adj = _r_adj.where(~_bad, np.nan)
                period_data["daily_ret"] = _r_adj
            else:
                if _mark == 'enforce':
                    period_data = period_data.sort_values(['ts_code', 'trade_date'])
                    period_data["daily_ret"] = period_data.groupby("ts_code")["close_p"].pct_change(fill_method=None)
                else:
                    period_data["daily_ret"] = period_data.groupby("ts_code")["close_p"].pct_change()
            if _mark == 'enforce':
                expected_dates = market_dates[(market_dates > entry_date) & (market_dates <= exit_date)]
                held = period_data.pivot(index='trade_date', columns='ts_code', values='daily_ret')
                held = held.reindex(index=expected_dates, columns=sorted(selected_codes))
                held = held.where(np.isfinite(held))
                missing = held.isna().sum(axis=1)
                _mark_diag['missing_held_price_cells'] += int(missing.sum())
                _mark_diag['incomplete_marking_days'] += int((missing > 0).sum())
                for year, count in missing.groupby(missing.index.year).sum().items():
                    counts = _mark_diag['missing_held_price_cells_by_year']
                    counts[str(year)] = counts.get(str(year), 0) + int(count)
                for day in missing[missing > 0].index:
                    if len(_mark_diag['marking_samples']) < 12:
                        _mark_diag['marking_samples'].append({'date': str(day.date()), 'missing': int(missing.loc[day])})
                # Missing quotes are unknown, not zero returns or redistributed capital.
                daily_portfolio = held.sum(axis=1, min_count=self.top_n) / self.top_n
                daily_portfolio.name = 'daily_ret'
            else:
                daily_portfolio = period_data.groupby("trade_date")["daily_ret"].mean()

            # P0-g: 成本在持有期首个有效收益日一次性扣除
            if _cost > 0 and len(daily_portfolio):
                _valid = daily_portfolio.dropna()
                if len(_valid):
                    daily_portfolio = daily_portfolio.copy()
                    _d0 = _valid.index[0]
                    daily_portfolio.loc[_d0] = daily_portfolio.loc[_d0] - _cost
            
            portfolio_returns.append(daily_portfolio)

        _set_diag(possible=n_possible, used=n_used, skipped=n_skipped,
                  samples=skipped_samples, blk_up=n_blk_up, blk_fb=n_blk_fb,
                  blk_dn=n_blk_dn, cost=total_cost, turns=turns,
                  adj_clipped=n_adj_clipped,
                  sparse=summarize_sparse_diag(sparse_periods, self.top_n))

        if not portfolio_returns:
            return _empty

        port_ret = pd.concat(portfolio_returns).sort_index()
        # 去重 (同一日期取平均)
        if _mark == 'enforce':
            if port_ret.index.has_duplicates:
                raise ValueError('overlapping holdings return periods')
        else:
            port_ret = port_ret.groupby(port_ret.index).mean()
        
        return port_ret, pd.Series(dtype=float)

    def _backtest_factor_fixed_shares(self, factor_df: pd.DataFrame
                                      ) -> Tuple[pd.Series, pd.Series]:
        """M11: 固定股数现金账本口径 (``FUXI_S5_LEDGER=fixed_shares``, 默认 off)。

        与 proxy 路径的**可声明差异** (对拍读数见 reports/fuxi_m11_ledger_parity_*):
          · 建仓 = 信号日次一交易日**开盘**成交 (proxy 以信号日收盘价为基准);
          · 持仓 = 固定股数, 不日频再平衡 ⇒ 权重自然漂移;
          · 费用 = 每笔成交 notional × bps (proxy 为换手 notional 持有期首日一次性);
          · 整手 lot (主板 100 / 科创 200 / 北交所 100), 零碎股按 floor;
          · 无涨跌停证据的订单递延次日 (defer), 不猜价格成交;
          · 除权日必须有事证 (actions); 未供 ⇒ fail-closed ⇒ data_insufficient,
            缺数清单 (reasons / missing) 可枚举; 未决陈旧 mark 不授 valid。

        🔴 账本未闭合 ⇒ 返回空序列, 由 ``_backtest_and_metrics`` 短路为
        ``ledger_data_insufficient`` (不产出经济裁决读数; 禁止以代理读数替代)。
        本路径不产生 S5 通过 / JQ 资格。PIT 事件库与退市证据为已知缺口 ⇒
        生产开启后大概率呈现 data_insufficient (正确行为, 非故障)。
        """
        from s5_ledger_bridge import fixed_shares_book, load_ledger_bars
        try:
            from portfolio_execution_ledger import LotRule
        except Exception as exc:  # noqa: BLE001
            self._last_backtest_diag = {
                "ledger_route": True, "ledger_status": "error",
                "ledger_reasons": [f"ledger_import_failed: {exc}"[:200]],
                "weighting_contract":
                    "fixed_share_cash_ledger_under_declared_assumptions",
                "formal_eligibility": False}
            return pd.Series(dtype=float), pd.Series(dtype=float)

        _gate = _s5_tradable_gate()
        _tie = _s5_tie_rule()
        _bps = _s5_cost_bps()
        _top_n = int(self.top_n)
        _empty = (pd.Series(dtype=float), pd.Series(dtype=float))

        def _emit(status, reasons, **extra):
            diag = {
                "ledger_route": True,
                "ledger_status": status,
                "ledger_reasons": [str(r)[:160] for r in reasons][:24],
                "weighting_contract":
                    "fixed_share_cash_ledger_under_declared_assumptions",
                "formal_eligibility": False,
                "switches": {"ledger": "fixed_shares", "tradable_gate": _gate,
                             "tie_rule": _tie, "cost_bps": _bps},
            }
            diag.update(extra)
            self._last_backtest_diag = diag

        if factor_df is None or len(factor_df) == 0:
            _emit("data_insufficient", ["empty_signal"])
            return _empty

        # 价格 (未复权真实价; universe 已裁) — 只取 signals 与价格的内连接
        price = self._price_df[self._price_df["ts_code"].isin(self._universe)]
        _pcols = [c for c in ("ts_code", "trade_date", "close", "amount")
                  if c in price.columns]
        merged = factor_df.merge(price[_pcols], on=["ts_code", "trade_date"],
                                 how="inner")
        if len(merged) == 0:
            _emit("data_insufficient", ["signal_price_join_empty"])
            return _empty

        # 调仓日 = 周最后交易日 (与 backtest_factor 同源: signals 实际覆盖的日历)
        calendar = merged[["trade_date"]].drop_duplicates().copy()
        calendar["week"] = calendar.trade_date.dt.to_period("W-SUN")
        week_last = calendar.groupby("week")["trade_date"].max().reset_index()
        rebalance_dates = sorted(week_last["trade_date"].unique())
        if len(rebalance_dates) < 2:
            _emit("data_insufficient", ["insufficient_rebalance_dates"])
            return _empty

        # 逐周选股 (语义与 backtest_factor 选择段保持同步; 改动需两处)
        _ul_wide = self._get_limit_up_wide() if _gate else None
        plan = []
        n_used = 0
        n_skipped = 0
        n_blk_up = 0
        skipped_samples = []
        prev_codes = None
        for _i in range(len(rebalance_dates) - 1):
            entry_date = rebalance_dates[_i]
            entry_data = merged[merged["trade_date"] == entry_date].copy()
            entry_data = entry_data[np.isfinite(
                pd.to_numeric(entry_data["factor_value"], errors="coerce"))]
            if _gate and len(entry_data):
                if _ul_wide is not None and entry_date in _ul_wide.index:
                    _ul = entry_data["ts_code"].map(_ul_wide.loc[entry_date])
                else:
                    _ul = pd.Series(np.nan, index=entry_data.index,
                                    dtype="float64")
                _hit_exact = _ul.notna() & (entry_data["close"].to_numpy()
                                            >= _ul.to_numpy() - 1e-6)
                if _hit_exact.any():
                    n_blk_up += int(np.sum(_hit_exact))
                    entry_data = entry_data[~_hit_exact]
            if len(entry_data) < _top_n:
                n_skipped += 1
                if len(skipped_samples) < 12:
                    skipped_samples.append({
                        "date": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                        "n_after_gate": int(len(entry_data)), "need": _top_n})
                # 非首期: 不给新 targets ⇒ 账本按无新指令延续原持仓 (等价 proxy
                # 的 carried_periods; 不引入隐性再平衡)。首期不足则不建仓。
                continue
            if _tie == "on":
                _by = [c for c in ("factor_value", "amount", "ts_code")
                       if c in entry_data.columns]
                _asc = [False if c in ("factor_value", "amount") else True
                        for c in _by]
                entry_data = entry_data.sort_values(_by, ascending=_asc,
                                                    kind="mergesort")
            elif _tie == "stable":
                entry_data = entry_data.sort_values("factor_value",
                                                    ascending=False,
                                                    kind="mergesort")
            else:
                entry_data = entry_data.sort_values("factor_value",
                                                    ascending=False)
            target_codes = sorted(str(c) for c in
                                  entry_data.head(_top_n)["ts_code"])
            prev_codes = set(target_codes)
            n_used += 1
            plan.append((pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                         {c: 1.0 / len(target_codes) for c in target_codes}))

        if not plan:
            _emit("data_insufficient", ["no_rebalance_periods"],
                  n_rebalance_skipped=n_skipped)
            return _empty

        all_codes = sorted({c for _, w in plan for c in w})
        lo = pd.Timestamp(merged.trade_date.min()).strftime("%Y-%m-%d")
        hi = pd.Timestamp(merged.trade_date.max()).strftime("%Y-%m-%d")
        try:
            days, bars, coverage = load_ledger_bars(all_codes, lo, hi,
                                                    raw_dir=_LEDGER_RAW_DIR)
        except Exception as exc:  # noqa: BLE001
            _emit("error", [f"bars_load_failed: {exc}"])
            return _empty
        # 账本日历 = 全 universe 市场日历 (codes 子集缺日不改变日历, 缺报价
        # 的 (日,股) 由 bars 返回 unknown ⇒ 持仓遇到即 fail-closed, 诚实)。
        _mc = self._price_df[self._price_df.trade_date.between(lo, hi)]
        market_days = sorted(pd.Timestamp(d).strftime("%Y-%m-%d")
                             for d in _mc.trade_date.unique())
        calendar_gap = sorted(set(market_days) - set(days))
        if market_days:
            days = market_days
        if not days:
            _emit("data_insufficient", ["empty_ledger_calendar"])
            return _empty

        rules = {}
        for c in all_codes:
            if str(c).startswith(("688", "689")):
                rules[c] = LotRule(200, 1)
            elif str(c).endswith(".BJ"):
                rules[c] = LotRule(100, 1)
            else:
                rules[c] = LotRule()

        # 声明模型情景 (首版即报告口径; 变更须走预注册):
        # floor 零碎股 / 无股息税情景 / 退市末价计价 / defer 递延 / vacuous 停牌。
        # actions 为空 = PIT 事件库缺口 ⇒ 除权未供事件即 fail-closed (预期行为)。
        book = fixed_shares_book(
            days, bars, dict(plan), initial_cash=10_000_000.0, cost_bps=_bps,
            rules=rules, actions=(), fractional_share_policy="floor",
            gross_cash_tax_rate=0.0, delist_mark_fraction=1.0,
            missing_limit_policy="defer",
            halted_session_check_policy="vacuous",
            unresolved_action_policy="block")

        _common = {"ledger_coverage": coverage,
                   "ledger_calendar_gap": calendar_gap[:24],
                   "n_rebalance_used": n_used,
                   "n_rebalance_skipped": n_skipped,
                   "n_rebalance_possible": len(rebalance_dates) - 1,
                   "n_blocked_limit_up": n_blk_up,
                   "skipped_samples": skipped_samples}
        status = book.get("status")
        if status != "book_complete_under_declared_assumptions":
            _emit("error" if status == "error" else "data_insufficient",
                  book.get("reasons") or ["ledger_incomplete"],
                  ledger_missing=book.get("missing"), **_common)
            return _empty

        rets = book["daily_returns"] or []
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rets])
        vals = [float(r) if r is not None else np.nan for _, r in rets]
        port_ret = pd.Series(vals, index=idx).dropna()
        _emit("valid", [],
              ledger_missing=book.get("missing"),
              ledger_total_fees=book.get("total_fees"),
              ledger_fees_frac=book.get("total_fees_frac"),
              ledger_n_fills=book.get("n_fills"),
              ledger_n_orders=book.get("n_orders"),
              n_periods_possible=len(rebalance_dates) - 1,
              n_periods_used=n_used, n_periods_skipped=n_skipped,
              coverage_rate=1.0, coverage_ok=True, **_common)
        return port_ret, pd.Series(dtype=float)

    def rank_sum_combo_returns(
        self,
        formulas: List[str],
        *,
        top_n: int = None,
        names: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        v0.6.1: 多因子等权 rank 和组合收益序列 (配对增量实验的 control/treatment 源)。

        对每个公式 evaluate_factor_formula → 按交易日横截面 rank(pct) → 等权平均
        合成单一因子 → backtest_factor 周频 top-N 回测。返回组合日收益 Series。

        任一公式评估失败不影响整体 (失败公式被跳过)。
        """
        if not formulas:
            return pd.Series(dtype=float)
        top_n = top_n or self.top_n
        names = names or [f"f{i}" for i in range(len(formulas))]
        frames = []
        for i, f in enumerate(formulas):
            if not f:
                continue
            try:
                fdf = self.evaluate_factor_formula(f, names[i] if i < len(names) else f"f{i}")
                if fdf is None or len(fdf) == 0:
                    continue
                # 横截面 rank(pct) (pandas 默认按列即按 ts_code 分组? 需 pivot 后 axis=1)
                piv = fdf.pivot_table(index="trade_date", columns="ts_code",
                                      values="factor_value")
                ranked = piv.rank(axis=1, pct=True)  # 宽表横截面 rank (铁律: axis=1)
                long = ranked.stack().rename("factor_value").reset_index()
                frames.append(long)
            except Exception:
                continue
        if not frames:
            return pd.Series(dtype=float)
        combo = pd.concat(frames).groupby(["trade_date", "ts_code"])["factor_value"].mean().reset_index()
        port_ret, _ = self.backtest_factor(combo)
        return port_ret

    @staticmethod
    def _calmar_mode() -> str:
        """Calmar 裁决口径 (P-20260912-003)。

        - ``vol_scaled``    : **默认**。分母由 |maxDD| 换成年化波动率 (IR 型),
                              天然与窗口长度无关 ⇒ 2025/2026 可比。
        - ``legacy``        : 分年全长口径 (2025 全年 vs 2026 至数据末端) — **不可比**,
                              仅供复现历史裁决 / A-B 对照。
        - ``common_window`` : 把 2025 截断到与 2026 相同日历窗口。**注意: 这是放宽**
                              (丢掉 2025 Q4), 仅作诊断, 不建议作默认。
        """
        _m = (os.environ.get("FUXI_S5_CALMAR_MODE", "vol_scaled") or "").strip().lower()
        return _m if _m in ("legacy", "common_window", "vol_scaled") else "vol_scaled"

    @staticmethod
    def _ir_threshold() -> float:
        """vol_scaled 口径下的 IR 阈值 (默认 0.7)。

        标定依据 — 蒙特卡洛 ``_calmar_montecarlo_20260912.py``, 固定过程
        (日漂移 +0.04% / 日波动 1.2% ⇒ 年化 10.6%), 每窗口 20000 条路径,
        且**严格复刻本类的算式** (ann_ret = total*PPY/len, 非几何年化):

            窗口      60     120    159    210    243    252
            legacy   3.148  1.766  1.467  1.191  1.132  1.090   ← 随窗口长度漂移 2.89x
            IR       0.531  0.539  0.550  0.541  0.564  0.558   ← 跨窗口平稳 (span 0.033)

        ⇒ ① legacy 的口径偏差是**实证事实**, 且偏差方向是**抬高短窗**
        (159 日 vs 243 日: 1.467 / 1.132 = 1.30x ⇒ 2026 的不完整年被高估);
        ② IR 无此偏差, 故为默认口径。

        阈值取 0.7 (而非均值对齐的 ~0.55) 的理由 — **假阳性率对齐**:
        在真实窗口上 legacy>1.0 的纯噪声通过率为 2025(243日) 0.393 /
        2026(159日) 0.440; IR 需 0.77 / 0.66 才能复现该通过率。
        取 0.7 ⇒ FP 0.416 / 0.428, 两侧都不比旧门槛松。若取 0.5 则 FP ≈ 0.485,
        **比旧门槛更宽松**, 与用户「不得放宽 S5」的立场冲突。
        故默认 0.7 = 口径修正而非放宽; 调高更严, 调低需显式声明。
        """
        try:
            return float(os.environ.get("FUXI_S5_IR_THRESHOLD", "0.7"))
        except (TypeError, ValueError):
            return 0.7

    def _effective_calmar_threshold(self) -> float:
        """当前口径下实际生效的 Calmar/IR 阈值。"""
        return self._ir_threshold() if self._calmar_mode() == "vol_scaled" \
            else float(self.calmar_threshold)

    @staticmethod
    def _annualized_calmar(port: pd.Series, mode: str, ppy: int) -> float:
        """按口径模式算「年化收益 / 风险分母」。"""
        if len(port) == 0:
            return 0.0
        ann_ret = ((1 + port).prod() - 1) * ppy / len(port)
        if mode == "vol_scaled":
            vol = port.std() * np.sqrt(ppy)
            return float(ann_ret / vol) if vol and vol > 0 else 0.0
        cum = (1 + port).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        return float(ann_ret / abs(mdd)) if mdd != 0 else 0.0

    def compute_metrics(
        self,
        portfolio_returns: pd.Series,
        year: int,
        end_monthday: str = None,
        mode: str = None,
    ) -> Dict[str, float]:
        """
        计算给定年份的组合指标。

        Args:
          end_monthday: ``"MM-DD"`` — 只取 ``year-01-01 ~ year-MM-DD`` 的窗口。
                        P-20260912-003: 用于让不完整年度与完整年度做到**同窗口可比**。
          mode:         Calmar 口径 (None = 读 env): legacy | common_window | vol_scaled

        Returns:
          {total_return, benchmark_return, excess_return, calmar, max_drawdown,
           annual_vol, periods_per_year, n_periods, calmar_full}
        """
        mode = mode or self._calmar_mode()
        _empty = {"total_return": 0, "benchmark_return": 0, "excess_return": 0,
                  "calmar": 0, "max_drawdown": 0, "annual_vol": 0,
                  "periods_per_year": 252, "n_periods": 0, "calmar_full": 0,
                  "calmar_mode": mode, "window_end": ""}
        aligned_dates = _s5_return_alignment()
        if portfolio_returns.empty:
            return dict(_empty)
        if self._index_df is None:
            return {**_empty, **({'error': 'benchmark_dates_missing'} if aligned_dates else {})}

        # v0.8 自动频次判断: 从完整收益序列推断年化因子 (防御周频/月频数据源)
        ppy = infer_periods_per_year(portfolio_returns.index)

        # 窗口末端: 同窗口口径下 2025 截断到 2026 的同一月日
        _end = f"{year}-12-31" if not end_monthday else f"{year}-{end_monthday}"

        # 过滤年份 (窗口)
        port_year = portfolio_returns[
            (portfolio_returns.index >= f"{year}-01-01") &
            (portfolio_returns.index <= _end)
        ].copy()
        if aligned_dates:
            port_year = port_year.replace([np.inf, -np.inf], np.nan).dropna()

        # 最小观测守卫: 按频次自适应 (月频一年12观测不误杀; 日频保持>=50)
        min_obs = max(6, min(50, ppy // 2))
        if len(port_year) < min_obs:
            r = dict(_empty)
            r["periods_per_year"] = ppy
            return r

        # 组合总收益
        total_return = (1 + port_year).prod() - 1

        # 基准收益
        idx_year = self._index_df[
            (self._index_df["trade_date"] >= f"{year}-01-01") &
            (self._index_df["trade_date"] <= _end)
        ].copy()

        if aligned_dates:
            matched = idx_year.drop_duplicates('trade_date').set_index('trade_date')['idx_return']
            matched = matched.reindex(port_year.index)
            if matched.isna().any():
                return {**_empty, 'error': 'benchmark_dates_missing'}
            benchmark_return = (1 + matched).prod() - 1
        elif len(idx_year) > 0:
            benchmark_return = (1 + idx_year["idx_return"].dropna()).prod() - 1
        else:
            benchmark_return = 0.0

        excess_return = total_return - benchmark_return

        # Max Drawdown
        cumulative = (1 + port_year).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Annual volatility (年化因子按实际数据频次)
        annual_vol = port_year.std() * np.sqrt(ppy)

        # Calmar (年化收益 / 最大回撤; vol_scaled 模式下分母改为年化波动率)
        calmar = self._annualized_calmar(port_year, mode, ppy)

        # 全长口径诊断: 仅在发生截断时另算, 供复评对照 (否则等于自身)
        calmar_full = calmar
        if end_monthday:
            full_year = portfolio_returns[
                (portfolio_returns.index >= f"{year}-01-01") &
                (portfolio_returns.index <= f"{year}-12-31")
            ]
            if len(full_year) >= min_obs:
                calmar_full = self._annualized_calmar(full_year, mode, ppy)

        return {
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "annual_vol": annual_vol,
            "periods_per_year": ppy,
            "n_periods": int(len(port_year)),
            "calmar_full": calmar_full,
            "calmar_mode": mode,
            "window_start": str(port_year.index.min().date()),
            "window_end": str(port_year.index.max().date()) if aligned_dates else _end,
            "window_end_requested": _end,
        }

    def _daily_ic_series(
        self,
        factor_df,
        forward_period: int = 5,
        min_stocks_per_day: int = 30,
    ):
        """从已评估的 factor_df 计算逐日 Rank IC 序列。

        v0.9.2 P-20260823-005: 供 DecayMonitor 真实数据回填 —
        此前 IC 缓存只有合成残留 (d1/d2/s1), 真实因子从未写入。
        """
        if factor_df is None or len(factor_df) == 0:
            return []
        price_data = self._price_df[self._price_df["ts_code"].isin(self._universe)][
            ["ts_code", "trade_date", "close_p"]
        ].drop_duplicates(subset=["ts_code", "trade_date"])
        merged = factor_df.merge(price_data, on=["ts_code", "trade_date"], how="inner")
        merged = merged.sort_values(["ts_code", "trade_date"])
        merged["fwd_return"] = merged.groupby("ts_code")["close_p"].shift(-forward_period) / merged["close_p"] - 1
        merged = merged.dropna(subset=["factor_value", "fwd_return"])
        ic_series = []
        for date, group in merged.groupby("trade_date"):
            if len(group) < min_stocks_per_day:
                continue
            rank_factor = group["factor_value"].rank()
            rank_fwd = group["fwd_return"].rank()
            ic = rank_factor.corr(rank_fwd)
            if not np.isnan(ic):
                ic_series.append(ic)
        return ic_series

    def compute_rank_ic(
        self,
        formula: str,
        factor_name: str = "unknown",
        forward_period: int = 5,
        min_stocks_per_day: int = 30,
    ) -> Dict[str, float]:
        """
        计算因子的 Rank IC (Spearman) 和 ICIR。
        
        用于 Stage 1 快速 IC 筛选。基于 daily_prices.csv 真实数据计算。
        
        Returns:
          {ic, icir, n_days, factor_coverage}
        """
        import warnings
        warnings.filterwarnings("ignore")
        
        if not self._loaded:
            self._load_data()

        if self._price_df is None:
            return {"ic": 0.0, "icir": 0.0, "n_days": 0, "factor_coverage": 0.0}

        # 评估因子公式
        factor_df = self.evaluate_factor_formula(formula, factor_name)
        if factor_df is None or len(factor_df) == 0:
            return {"ic": 0.0, "icir": 0.0, "n_days": 0, "factor_coverage": 0.0}
        
        # 构建价格 pivot 用于计算前向收益
        price_data = self._price_df[self._price_df["ts_code"].isin(self._universe)][
            ["ts_code", "trade_date", "close_p"]
        ].drop_duplicates(subset=["ts_code", "trade_date"])
        
        # Merge factor values with price data
        merged = factor_df.merge(price_data, on=["ts_code", "trade_date"], how="inner")
        
        # 计算前向收益
        merged = merged.sort_values(["ts_code", "trade_date"])
        merged["fwd_return"] = merged.groupby("ts_code")["close_p"].shift(-forward_period) / merged["close_p"] - 1
        
        # 移除 NaN
        merged = merged.dropna(subset=["factor_value", "fwd_return"])
        
        # 逐日计算 Rank IC
        ic_series = []
        for date, group in merged.groupby("trade_date"):
            if len(group) < min_stocks_per_day:
                continue
            # Rank IC (Spearman)
            rank_factor = group["factor_value"].rank()
            rank_fwd = group["fwd_return"].rank()
            ic = rank_factor.corr(rank_fwd)
            if not np.isnan(ic):
                ic_series.append(ic)
        
        if not ic_series:
            return {"ic": 0.0, "icir": 0.0, "n_days": 0, "factor_coverage": 0.0}
        
        ic_arr = np.array(ic_series)
        mean_ic = float(np.nanmean(ic_arr))
        std_ic = float(np.nanstd(ic_arr))
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        
        return {
            "ic": round(mean_ic, 6),
            "icir": round(icir, 4),
            "n_days": len(ic_series),
            "factor_coverage": round(len(merged) / len(factor_df), 4) if len(factor_df) > 0 else 0.0,
        }

    # ══════════════════════════════════════════════════════════════
    # P-20260910-008 周频链路修复: 执行层周频数据通道
    # ══════════════════════════════════════════════════════════════

    def _get_weekly_panel(self):
        """懒构建周频宽表面板 (由已加载日频面板重采样, W-SUN 标签)。

        与 weekly_lane 的周频环境字段对齐 (close/open/high/low/volume/amount/
        returns/vwap/hl_ratio/momentum_12_1/vol_60/log_size), 因此 S1 的
        rolling(4)=4 周在 S5 仍然是 4 周。
        """
        if self.weekly_channel == "off":
            return None
        if self._weekly_panel is not None:
            return self._weekly_panel
        try:
            _here = str(Path(__file__).parent)
            if _here not in sys.path:
                sys.path.insert(0, _here)
            from weekly_panel import build_weekly_panel
        except Exception as e:  # noqa: BLE001
            print(f"  [S5-W] ⚠️ weekly_panel 模块不可用, 保留原频率契约: {e}")
            return None
        try:
            _cols = ["ts_code", "trade_date", "open", "high", "low",
                     "close", "volume", "amount"]
            d = self._price_df[[c for c in _cols if c in self._price_df.columns]]
            print(f"  [S5-W] 构建周频面板 (源: 日频面板 {len(d):,} 行)...")
            self._weekly_panel = build_weekly_panel(d, verbose=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [S5-W] ⚠️ 周频面板构建失败, 保留原频率契约: {e}")
            self._weekly_panel = None
        return self._weekly_panel

    def evaluate_factor_formula_weekly(
        self, formula: str, factor_name: str = "unknown"
    ) -> Tuple[Optional[pd.DataFrame], Dict, str]:
        """周频口径求值 → 日频对齐长表。

        Returns
        -------
        (daily_long | None, weekly_ic_dict, reason)
          daily_long: [ts_code, trade_date, factor_value] — 周信号已对齐到
                      周内最后交易日并 ffill 至日频, 可直接喂 backtest_factor。
        """
        if len([line for line in formula.splitlines() if line.strip()]) <= 1:
            formula = self._normalize_rank_axis(formula)
        panel = self._get_weekly_panel()
        if panel is None:
            return None, {}, "周频面板不可用"
        try:
            from weekly_panel import (eval_weekly_formula, align_weekly_to_daily,
                                      weekly_ic as _weekly_ic_fn)
        except Exception as e:  # noqa: BLE001
            return None, {}, f"weekly_panel 导入失败: {e}"

        ok, wide, long, reason = eval_weekly_formula(formula, panel)
        if not ok:
            return None, {}, reason
        ic = _weekly_ic_fn(panel, wide, min_stocks=self.weekly_min_stocks)

        if self.sample_stocks is None:
            keys = self._price_df[["ts_code", "trade_date"]]
        else:
            keys = self._price_df[
                self._price_df["ts_code"].isin(self._universe)
            ][["ts_code", "trade_date"]]
        daily = align_weekly_to_daily(long, panel, keys)
        if daily is None or len(daily) == 0:
            return None, ic, "周频信号对齐日频后为空"
        return daily, ic, ""

    def _value_guards(self, factor_df: pd.DataFrame) -> str:
        """常数因子 + 价格水平代理拦截。返回 reason (空串=通过)。

        v0.7 P-025 原逻辑 (频次无关), 抽出以便日频/周频两条口径复用。
        """
        try:
            fvals = pd.to_numeric(factor_df["factor_value"], errors="coerce").dropna()
            if len(fvals) == 0:
                return "因子值全为 NaN"
            n_unique = fvals.round(12).nunique()
            if n_unique <= 5:
                return (f"🧹 常数因子拦截: 有效值仅 {n_unique} 个离散水平, "
                        f"选股无区分度")
        except Exception:  # noqa: BLE001
            pass

        try:
            price_col = self._price_df[
                self._price_df["ts_code"].isin(self._universe)
            ][["ts_code", "trade_date", "close"]].copy()
            merged = factor_df.merge(price_col, on=["ts_code", "trade_date"],
                                     how="inner")
            if len(merged) > 200:
                piv_f = merged.pivot_table(index="trade_date", columns="ts_code",
                                           values="factor_value")
                piv_c = merged.pivot_table(index="trade_date", columns="ts_code",
                                           values="close")
                corrs = piv_f.rank(axis=1).T.corrwith(piv_c.rank(axis=1).T)
                med = corrs.abs().median()
                if pd.notna(med) and med > 0.95:
                    return (f"🧹 价格水平代理拦截: 因子与收盘价 rank 相关 "
                            f"中位数 {med:.3f} > 0.95 (非 alpha)")
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _backtest_and_metrics(self, factor_df: pd.DataFrame):
        """因子长表 → (组合日收益 Series, 2025 指标, 2026 指标)。

        P-20260912-003: 2025(完整年) 与 2026(至当前月末) 的 maxDD 因窗口长度不同
        而不可比 ⇒ 按 ``FUXI_S5_CALMAR_MODE`` 统一口径后再裁决。
        ``common_window`` 下把 2025 截断到与 2026 相同的日历窗口 (同月份/同交易日数)。
        """
        port_ret, _ = self.backtest_factor(factor_df)
        _led_diag = self._last_backtest_diag or {}
        if _led_diag.get("ledger_status") in ("data_insufficient", "error"):
            # M11: 账本未闭合 ⇒ 不产出经济读数。禁止以代理读数替代, 也不得让
            # 空序列走 _verdict 被归因成"两项均不达标"(那是错归因: 真因是可枚举
            # 的缺数/事件未解析)。缺数清单由 ledger_reasons/missing 携带。
            _err = {
                "error": "ledger_data_insufficient",
                "evaluation_status": "data_insufficient",
                "diagnostic_only": False,
                "excess_return": float("nan"), "calmar": float("nan"),
                "total_return": float("nan"),
                "benchmark_return": float("nan"),
                "max_drawdown": float("nan"), "annual_vol": float("nan"),
                "n_periods": 0, "calmar_mode": self._calmar_mode(),
                "calmar_full": float("nan"),
                "ledger_reasons": list(_led_diag.get("ledger_reasons") or [])[:12],
            }
            return port_ret, dict(_err), dict(_err)
        mode = self._calmar_mode()
        end_md = None

        if mode == "common_window":
            try:
                _r26 = port_ret[port_ret.index >= "2026-01-01"]
                if len(_r26) > 0:
                    _last = _r26.index.max()
                    end_md = f"{_last.month:02d}-{_last.day:02d}"
            except Exception:  # noqa: BLE001
                end_md = None

        m25 = self.compute_metrics(port_ret, 2025, end_monthday=end_md, mode=mode)
        m26 = self.compute_metrics(port_ret, 2026, mode=mode)
        diag = self._last_backtest_diag or {}
        for year, metrics in ((2025, m25), (2026, m26)):
            missing = diag.get('missing_held_price_cells_by_year', {}).get(str(year), 0)
            if missing:
                metrics.update({'error': 'held_price_observations_missing', 'evaluation_status': 'data_insufficient',
                                'missing_held_price_cells': missing,
                                'diagnostic_only': True})
        return port_ret, m25, m26

    def _verdict(self, m25: Dict, m26: Dict, label: str = "") -> Tuple[bool, str]:
        """S5 判定 (两年超额 + 两年 Calmar)。"""
        if m25.get('error') or m26.get('error'):
            return False, '不可裁决: ' + str(m25.get('error') or m26.get('error'))
        e25, e26 = m25["excess_return"], m26["excess_return"]
        c25, c26 = m25["calmar"], m26["calmar"]
        _cm = m25.get("calmar_mode", "legacy")
        _th = self._effective_calmar_threshold()
        _tag = ("" if _cm == "legacy"
                else f" [口径={_cm}{'/IR' if _cm == 'vol_scaled' else ''},阈值={_th:g}]")
        excess_ok = bool(e25 > self.excess_threshold) and bool(e26 > self.excess_threshold)
        calmar_ok = bool(c25 > _th) and bool(c26 > _th)
        passed = bool(excess_ok and calmar_ok)
        if passed:
            reason = (f"✅ 两年正向过滤通过: excess_25={e25:.2%}, excess_26={e26:.2%}, "
                      f"calmar_25={c25:.2f}, calmar_26={c26:.2f}{_tag}")
        elif excess_ok and not calmar_ok:
            reason = (f"❌ Calmar不达标: calmar_25={c25:.2f}, calmar_26={c26:.2f} "
                      f"(需 >{_th:g}){_tag}")
        elif not excess_ok and calmar_ok:
            reason = (f"❌ 超额不达标: excess_25={e25:.2%}, excess_26={e26:.2%} "
                      f"(需 >{self.excess_threshold:.0%}){_tag}")
        else:
            reason = (f"❌ 两项均不达标: excess_25={e25:.2%}, excess_26={e26:.2%}, "
                      f"calmar_25={c25:.2f}, calmar_26={c26:.2f}{_tag}")
        if label:
            reason = f"[{label}] {reason}"
        return passed, reason

    @staticmethod
    def _fill_metrics(result: "S5Result", m25: Dict, m26: Dict) -> None:
        """把分年度指标写入 S5Result 主字段。"""
        result.excess_2025 = m25["excess_return"]
        result.excess_2026 = m26["excess_return"]
        result.calmar_2025 = m25["calmar"]
        result.calmar_2026 = m26["calmar"]
        result.total_return_2025 = m25["total_return"]
        result.total_return_2026 = m26["total_return"]
        result.benchmark_return_2025 = m25["benchmark_return"]
        result.benchmark_return_2026 = m26["benchmark_return"]
        result.max_drawdown_2025 = m25["max_drawdown"]
        result.max_drawdown_2026 = m26["max_drawdown"]
        result.annual_vol_2025 = m25["annual_vol"]
        result.annual_vol_2026 = m26["annual_vol"]
        # P-20260912-003 口径诊断字段
        result.calmar_mode = m25.get("calmar_mode", "legacy")
        result.calmar_2025_full = m25.get("calmar_full", m25["calmar"])
        result.calmar_2026_full = m26.get("calmar_full", m26["calmar"])
        result.window_days_2025 = m25.get("n_periods", 0)
        result.window_days_2026 = m26.get("n_periods", 0)
        result.calmar_window_matched = bool(
            result.window_days_2025 and result.window_days_2026
            and abs(result.window_days_2025 - result.window_days_2026) <= 3)

    def _settle_weekly_shadow(self, result: "S5Result", weekly_df: pd.DataFrame) -> None:
        """周频口径独立结算 → 写入 result.weekly_* (不改 result.passed)。

        与日频口径解耦: 周频种子常引用日频面板不存在的字段 (momentum_12_1 /
        vol_60 / log_size), 日频口径会直接 eval 失败, 但周频口径仍应留痕。
        """
        try:
            _wp, _w25, _w26 = self._backtest_and_metrics(weekly_df)
            _w_pass, _w_reason = self._verdict(_w25, _w26, "周频口径")
            result.weekly_passed = _w_pass
            result.weekly_reason = _w_reason
            result.weekly_excess_2025 = _w25["excess_return"]
            result.weekly_excess_2026 = _w26["excess_return"]
            result.weekly_calmar_2025 = _w25["calmar"]
            result.weekly_calmar_2026 = _w26["calmar"]
        except Exception as _we:  # noqa: BLE001
            result.weekly_reason = (f"周频口径回测失败: {type(_we).__name__}: "
                                    f"{str(_we)[:80]}")

    def validate_factor(
        self,
        factor_name: str,
        formula: str,
        skip_simplify: bool = False,
        natural_freq: str = "daily",
    ) -> S5Result:
        """
        对单个因子执行完整 S5 验证。

        v0.6.1: skip_simplify=True 时跳过规则6 (过度简化拦截), 用于 seed 重检。

        P-20260910-008 (周频链路修复): natural_freq=="weekly" 时按
        FUXI_S5_WEEKLY_CHANNEL 走周频数据通道 —
          - enforce: 用周频口径 (rolling(N) = N 周) 裁决
          - shadow : 周频口径照算并落盘审计, 裁决仍用日频口径 (零副作用)
          - off    : 修复前行为 (周频公式按日频求值)
        enforce 下周频通道异常记不可评估，不得替换为日频；shadow 保留兼容回退。

        Returns:
          S5Result with passed=True if:
            - excess_2025 > 0 AND excess_2026 > 0
            - calmar_2025 > 1.0 AND calmar_2026 > 1.0
        """
        result = S5Result(factor_name=factor_name, formula=formula[:200])
        t0 = datetime.now()
        # v0.7.23 P0-h: 清空上一次回测诊断, 防早期 return 时读到陈旧值
        self._last_backtest_diag = None
        _freq = "weekly" if str(natural_freq or "").lower() == "weekly" else "daily"
        result.weekly_channel_mode = self.weekly_channel if _freq == "weekly" else "off"

        if not self.is_ready:
            result.error = "Price data not available"
            result.reason = "价格数据不可用"
            return result

        try:
            self._load_data()
        except Exception as e:
            result.error = str(e)
            result.reason = f"数据加载失败: {e}"
            return result

        # v0.4: 公式合理性检查（拦截垃圾因子）
        # v0.7.19: 加载器公式 (裸标识符) 不是 pandas 表达式, 规则6「过度简化」
        # 必然误杀 (op_count=0) → 豁免; 其有效性由回测裁决, 不靠公式形态。
        _is_loader = (_loader_formula_switch() == "on"
                      and (formula or "").strip() in LOADER_FORMULA_REGISTRY)
        sanity_reason = self._check_formula_sanity(
            formula, factor_name,
            skip_simplify=(skip_simplify or _is_loader))
        if sanity_reason:
            result.reason = f"🧹 公式合理性拦截: {sanity_reason}"
            return result

        # ── P-20260910-008: 分频求值 ────────────────────────
        # shadow 模式下周频口径照算 (审计留痕), 仅 enforce 模式用它裁决。
        _weekly_df = None
        _weekly_ic = {}
        if _freq == "weekly" and self.weekly_channel in ("shadow", "enforce"):
            _weekly_df, _weekly_ic, _werr = self.evaluate_factor_formula_weekly(
                formula, factor_name)
            if _weekly_ic:
                result.weekly_ic = float(_weekly_ic.get("weekly_ic", 0.0) or 0.0)
                result.weekly_icir = float(_weekly_ic.get("weekly_icir", 0.0) or 0.0)
                result.weekly_n = int(_weekly_ic.get("weekly_n", 0) or 0)
            if _weekly_df is None:
                if result.weekly_channel_mode == 'enforce':
                    result.signal_freq = 'weekly'
                    result.error = 'weekly_data_unavailable'
                    result.reason = f'周频契约不可评估: {_werr}; 禁止替换为日频'
                    return result
                print(f"    [S5-W] {factor_name}: 周频通道失败 ({_werr}), 回退日频口径")
                result.weekly_channel_mode = "fallback_daily"

        _use_weekly = (_freq == "weekly"
                       and self.weekly_channel == "enforce"
                       and _weekly_df is not None)

        # shadow 模式: 周频口径独立结算 (即使日频口径 eval 失败也留痕)
        if (not _use_weekly and self.weekly_channel == "shadow"
                and _weekly_df is not None):
            self._settle_weekly_shadow(result, _weekly_df)

        # 评估「裁决口径」的因子公式 (enforce=周频; 其余=日频)
        factor_df = (_weekly_df if _use_weekly
                     else self.evaluate_factor_formula(formula, factor_name))
        if factor_df is None or len(factor_df) == 0:
            result.reason = "因子公式评估无有效数据"
            return result
        result.signal_freq = "weekly" if _use_weekly else "daily"

        # v0.7 P-025: 常数/低基数 + 价格水平代理拦截 (频次无关, 两条口径共用)
        guard_reason = self._value_guards(factor_df)
        if guard_reason:
            result.reason = guard_reason
            return result

        # v0.9.2 P-20260823-005: 逐日 IC 序列写入 DecayMonitor 缓存
        # (修复: cache_ic_series 此前零调用, DecayMonitor 长期空转于合成残留数据)
        # P-20260910-008: 周频口径下不写日频 IC (语义不符), 仅日频口径写入。
        if not _use_weekly:
            try:
                ic_series = self._daily_ic_series(factor_df)
                if len(ic_series) >= 20:
                    from decay_monitor import get_decay_monitor
                    get_decay_monitor().update_ic_cache(factor_name, ic_series)
            except Exception:
                pass

        # 回测 + 分年度指标
        port_ret, metrics_2025, metrics_2026 = self._backtest_and_metrics(factor_df)
        # v0.6.1: 收益序列附回 (S6 DSR / 增量门禁用)
        result.portfolio_returns = port_ret
        result.portfolio_returns_n = len(port_ret)
        
        # 分年度指标 → 主字段
        self._fill_metrics(result, metrics_2025, metrics_2026)
        result.calmar_threshold_effective = self._effective_calmar_threshold()
        _marking = self._last_backtest_diag or {}
        result.marking_diagnostics = {
            "missing_held_price_cells": int(_marking.get("missing_held_price_cells", 0)),
            "missing_held_price_cells_by_year": dict(
                _marking.get("missing_held_price_cells_by_year", {}) or {}),
            "incomplete_marking_days": int(_marking.get("incomplete_marking_days", 0)),
            "marking_samples": list(_marking.get("marking_samples", []) or []),
            "weighting_contract": _marking.get("weighting_contract"),
        }
        result.economic_readings_status = (
            "diagnostic_only_data_insufficient"
            if metrics_2025.get('error') or metrics_2026.get('error') else "measured")
        if metrics_2025.get('error') or metrics_2026.get('error'):
            result.error = metrics_2025.get('error') or metrics_2026.get('error')
            if result.error == 'held_price_observations_missing':
                result.reason = '持仓行情缺失，禁止删股重加权或补零后裁决'
            elif result.error == 'ledger_data_insufficient':
                _leds = "；".join(str(x) for x in
                                  (metrics_2025.get('ledger_reasons') or [])[:6])
                result.reason = (f"账本口径未闭合 (data_insufficient): {_leds}"
                                 f"{' …' if len(metrics_2025.get('ledger_reasons') or []) > 6 else ''}"
                                 if _leds else
                                 "账本口径未闭合 (data_insufficient): 未列明原因")
            else:
                result.reason = '基准与组合实际收益日无法完整对齐'
            return result

        # S5 判定 (裁决口径)
        result.passed, result.reason = self._verdict(
            metrics_2025, metrics_2026, "周频口径" if _use_weekly else "")

        # ── P-20260910-008: 分频审计落填 ────────────────────
        if _use_weekly:
            result.weekly_passed = result.passed
            result.weekly_reason = result.reason
            result.weekly_excess_2025 = result.excess_2025
            result.weekly_excess_2026 = result.excess_2026
            result.weekly_calmar_2025 = result.calmar_2025
            result.weekly_calmar_2026 = result.calmar_2026
        elif self.weekly_channel == "shadow" and _weekly_df is not None:
            # shadow: 裁决 = 日频 (现行行为), 周频口径已在上方独立结算
            result.daily_caliber_passed = result.passed
            result.daily_caliber_reason = result.reason

        # ── v0.7.23 P0-h: 覆盖率披露 + 可选闸门 ─────────────────────
        # 覆盖不足的调仓期原本被静默整周丢弃 (不记录/不计入), 且
        # ann_ret = total*PPY/len 的自适应年化会掩盖少算的期数 ⇒ 读数不可归因。
        # 默认 FUXI_S5_MIN_COVERAGE=0 = 只披露不拦截 (零回归)。
        _diag = self._last_backtest_diag
        if isinstance(_diag, dict):
            result.coverage_rate = float(_diag.get("coverage_rate", 1.0) or 0.0)
            result.periods_skipped = int(_diag.get("n_periods_skipped", 0) or 0)
            result.coverage_ok = bool(_diag.get("coverage_ok", True))
            result.engine_switches = dict(_diag.get("switches", {}) or {})
            _need_cov = float(_diag.get("min_coverage_required", 0.0) or 0.0)
            if _need_cov > 0 and not result.coverage_ok:
                result.passed = False
                result.reason = (
                    f"❌ 覆盖率不达标 (P0-h): {result.coverage_rate:.1%} < "
                    f"{_need_cov:.1%} (跳过 {result.periods_skipped}/"
                    f"{int(_diag.get('n_periods_possible', 0) or 0)} 期) | "
                    f"原裁决: {result.reason}")

            # ── P-20260916-014a: 稀疏结构体检注记 (纯披露, 不改 passed) ──
            # 命中时把「IC 有效 ≠ TOP-N 有效」的原因显式写进 reason, 避免
            # 失败归因被误读为「因子无效」(实测该误读已发生两次)。
            _sd = _diag.get("sparse_diag")
            if isinstance(_sd, dict) and _sd.get("warning"):
                result.sparse_diag = _sd
                result.reason = f"{result.reason} | {_sd['warning']}"
            elif isinstance(_sd, dict):
                result.sparse_diag = _sd

        result.computation_time = (datetime.now() - t0).total_seconds()
        return result

    def validate_batch(
        self,
        candidates: List[Dict],
        verbose: bool = True,
    ) -> List[S5Result]:
        """
        批量验证多个候选因子。
        
        Parameters:
          candidates: [{factor_name, formula, ...}, ...]
        
        Returns:
          List of S5Result
        """
        results = []
        for i, cand in enumerate(candidates):
            name = cand.get("factor_name", f"candidate_{i}")
            formula = cand.get("formula", cand.get("expression", ""))
            
            if not formula:
                results.append(S5Result(
                    factor_name=name, passed=False,
                    reason="无公式表达式", error="No formula"
                ))
                continue

            if verbose:
                print(f"  [S5:{i+1}/{len(candidates)}] Validating {name[:40]}...")

            # v0.6.1: seed 重检候选跳过"过度简化"拦截 (原式天然简单且 JQ 已验证)
            skip_simplify = bool(cand.get("_seed_recheck"))
            # P-20260910-008: natural_freq 透传 → weekly 候选走周频数据通道
            _freq = str(cand.get("natural_freq") or "daily").strip().lower()
            result = self.validate_factor(name, formula, skip_simplify=skip_simplify,
                                          natural_freq=_freq)
            results.append(result)

            if verbose and result.passed:
                _tag = "[周频]" if result.signal_freq == "weekly" else ""
                print(f"    ✅ PASS{_tag}: excess_25={result.excess_2025:.2%} excess_26={result.excess_2026:.2%} "
                      f"calmar_25={result.calmar_2025:.2f} calmar_26={result.calmar_2026:.2f}")
            elif verbose and not result.passed:
                print(f"    {result.reason[:100]}")
                if result.weekly_passed is not None and result.signal_freq != "weekly":
                    print(f"    [周频口径对照] passed={result.weekly_passed}: "
                          f"{result.weekly_reason[:100]}")

        passed_count = sum(1 for r in results if r.passed)
        if verbose:
            print(f"\n  [S5] Batch result: {passed_count}/{len(results)} passed")

        return results


# ── 便捷函数 ──────────────────────────────────────────────

def get_s5_filter(**kwargs) -> S5JointFilter:
    """获取 S5 过滤器实例 (默认配置)"""
    return S5JointFilter(**kwargs)


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  S5 Joint Filter 测试")
    print("=" * 60)

    s5 = S5JointFilter(top_n=80, sample_stocks=200)

    if not s5.is_ready:
        print("⚠️ 价格数据不可用，跳过测试")
        sys.exit(0)

    # 测试几个因子
    test_factors = [
        ("overnight_momentum", "(open_p / close_p.shift(1) - 1).rolling(5).mean()"),
        ("intraday_reversal", "-(close_p / open_p - 1).rolling(5).mean()"),
        ("volume_stability", "-(volume_p.rolling(20).std() / volume_p.rolling(20).mean())"),
    ]

    results = s5.validate_batch(
        [{"factor_name": n, "formula": f} for n, f in test_factors],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  S5 验证结果汇总")
    print("=" * 60)
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.factor_name:30s}: "
              f"excess_25={r.excess_2025:+.2%} excess_26={r.excess_2026:+.2%} "
              f"| calmar_25={r.calmar_2025:.2f} calmar_26={r.calmar_2026:.2f} "
              f"| time={r.computation_time:.1f}s")
