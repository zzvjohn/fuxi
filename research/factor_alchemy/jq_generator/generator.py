# -*- coding: utf-8 -*-
"""
JQ Code Generator — 从因子组合自动生成聚宽策略代码

使用方式:
  from jq_generator import JQGenerator
  gen = JQGenerator(weights={'relative_spread_proxy': 0.358, 'gross_margin': 0.225})
  gen.generate('fa_v79_test_jq.py')

模板: 以 v7.6 均衡 (fa_v76_balanced_jq.py) 为骨架 — 唯一在 JQ 上正收益验证过的架构
  - get_price 逐字段 + 3路返回类型分发
  - ffill(价格) + fillna(0)(量)
  - 阈值股票池 (mcap >= N亿)
  - rank-percentile → rank-product (加权秩乘积，对齐 V2/V3 聚宽策略)
  - rank-product 加权秩乘积合成 (v8.0)
  - per-factor try/except 异常降级
  - get_current_data 停牌/涨停检查
  - 保护限价 (optional)
"""

import os
import datetime
from .registry import (
    JQ_FACTOR_REGISTRY, get_factor_jq_meta,
    get_required_price_fields, get_max_window,
)


class JQGenerator:
    """聚宽策略代码生成器"""

    def __init__(self, weights, name='auto_generated', mcap_min=5,
                 stock_num=30, max_turnover=0.60, use_protection_limit=False,
                 benchmark='000300.XSHG', rebalance_weekday=5, rebalance_time='14:50'):
        """
        Args:
            weights: dict {factor_name: weight} — 权重值 (可以是任意符号, 内部判断方向)
            name: 策略名称
            mcap_min: 流通市值下限 (亿)
            stock_num: 持仓数量
            max_turnover: 最大单边换手率
            use_protection_limit: 是否启用保护限价 (#12)
            benchmark: 基准指数
            rebalance_weekday: 调仓日 (5=周五)
            rebalance_time: 调仓时间
        """
        self.weights = weights
        self.name = name
        self.mcap_min = mcap_min
        self.stock_num = stock_num
        self.max_turnover = max_turnover
        self.use_protection_limit = use_protection_limit
        self.benchmark = benchmark
        self.rebalance_weekday = rebalance_weekday
        self.rebalance_time = rebalance_time

        self._validate_factors()

    def _validate_factors(self):
        """校验所有因子都在 registry 中"""
        missing = [f for f in self.weights if f not in JQ_FACTOR_REGISTRY]
        if missing:
            raise ValueError(
                "以下因子未在 JQ_FACTOR_REGISTRY 中定义, 请先添加映射:\n"
                + "\n".join(f"  - {f}" for f in missing)
            )
        unverified = [f for f in self.weights if not JQ_FACTOR_REGISTRY[f].get('verified')]
        if unverified:
            print("[WARN] 以下因子尚未经 JQ 交叉验证 (verified=False):")
            for f in unverified:
                print(f"  - {f} ({JQ_FACTOR_REGISTRY[f]['category']})")

    # ============================================================
    # 代码生成
    # ============================================================

    def generate(self, output_path=None):
        """生成完整 JQ 策略代码, 返回字符串; 若提供 output_path 则写入文件"""
        code = self._build_header()
        code += self._build_imports()
        code += self._build_params()
        code += self._build_initialize()
        code += self._build_main()
        code += self._build_stock_pool()
        code += self._build_load_price_data()
        code += self._build_factor_dispatch()
        code += self._build_factor_functions()
        code += self._build_utils()
        code += self._build_rebalance()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"[JQGenerator] 策略代码已写入: {output_path}")

        return code

    def _build_header(self):
        factor_list = "\n".join(
            f"  {f}  (w={w:.3f})   {JQ_FACTOR_REGISTRY[f]['direction_note'][:60]}"
            for f, w in self.weights.items()
        )
        date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        return f'''# -*- coding: utf-8 -*-
"""
================================================================
因子炼金术 JQ 自动生成 — {self.name}
================================================================
生成时间: {date_str}
生成器: jq_generator (P0.2)

因子组合:
{factor_list}

参数:
  持仓数={self.stock_num}, 市值下限={self.mcap_min}亿, 最大换手={self.max_turnover:.0%}
  调仓=周{self.rebalance_weekday} {self.rebalance_time}, 基准={self.benchmark}
  保护限价={'是' if self.use_protection_limit else '否 (直接市价单)'}

成本: QMT实盘 = 佣金万1 + 印花千1 + 滑点0.3%
      JQ FixedSlippage(0.003) = 0.003元/股, 低估真实成本
================================================================
"""
'''

    def _build_imports(self):
        return '''
from __future__ import division
import datetime
import numpy as np
import pandas as pd

'''

    def _build_params(self):
        # 生成 WEIGHTS 字典
        w_lines = ",\n".join(f"    '{f}': {w:.4f}" for f, w in self.weights.items())
        factor_names = list(self.weights.keys())
        factor_list_py = "[" + ", ".join(f"'{f}'" for f in factor_names) + "]"
        max_window = get_max_window(factor_names)
        price_fields = get_required_price_fields(factor_names)
        price_fields_py = str(price_fields)

        return f'''
# ==================== 策略参数 ====================
STOCK_NUM      = {self.stock_num}
MAX_TURNOVER   = {self.max_turnover:.2f}
MCAP_MIN       = {self.mcap_min}
EXCLUDE_PREFIX = ('8', '4')

# 因子权重 (由 JQGenerator 自动生成)
WEIGHTS = {{
{w_lines}
}}
FACTOR_NAMES = {factor_list_py}

# 数据加载参数
PRICE_FIELDS = {price_fields_py}
PRICE_WINDOW = {max_window + 5}   # 最大回溯 + 5天余量

DEBUG = True
USE_PROTECTION_LIMIT = {str(self.use_protection_limit)}

'''

    def _build_initialize(self):
        return f'''
# ==================== 初始化 ====================
def initialize(context):
    set_benchmark('{self.benchmark}')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_order_cost(OrderCost(
        close_tax=0.001,
        open_commission=0.0001,
        close_commission=0.0001,
        min_commission=5,
    ), type='stock')
    set_slippage(FixedSlippage(0.003))
    run_weekly(main, weekday={self.rebalance_weekday}, time='{self.rebalance_time}')

    g.stock_num      = STOCK_NUM
    g.max_turnover   = MAX_TURNOVER
    g.mcap_min       = MCAP_MIN
    g.exclude_prefix = EXCLUDE_PREFIX
    g.weights        = WEIGHTS
    g.factor_names   = FACTOR_NAMES
    g.price_fields   = PRICE_FIELDS
    g.price_window   = PRICE_WINDOW
    g.last_holdings  = set()
    g.debug          = DEBUG
    g.turnover_rate_hist = {{}}   # turnover_std 滚动历史缓存

'''

    def _build_main(self):
        return '''
# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')

    stocks = get_stock_pool(context)
    n_stocks = len(stocks)
    if g.debug:
        log.info('[%s] STEP1 pool=%d stocks' % (date_str, n_stocks))
    if n_stocks < g.stock_num:
        return

    price_data = load_price_data(stocks, context)

    factor_arrays = {}
    factor_valid = {}

    for fname in g.factor_names:
        try:
            arr, valid = compute_factor(fname, stocks, context, price_data)
            factor_arrays[fname] = arr
            factor_valid[fname] = valid
            n_valid = int(np.sum(valid))
            if n_valid == 0:
                # 全 NaN: 占位符/数据缺失/逻辑错误的强提醒, 不再静默
                log.error('[%s] STEP2 %s: valid=0/%d (0.0%%) <<< 因子完全塌缩! 检查 jq_compute_function 是否为占位符或价格数据缺失' % (
                    date_str, fname, n_stocks))
            elif g.debug:
                log.info('[%s] STEP2 %s: valid=%d/%d (%.1f%%)' % (
                    date_str, fname,
                    n_valid, n_stocks,
                    100 * n_valid / max(n_stocks, 1)))
        except Exception as e:
            log.error('[%s] STEP2 %s FAILED: %s' % (date_str, fname, str(e)))
            factor_arrays[fname] = np.full(n_stocks, np.nan)
            factor_valid[fname] = np.zeros(n_stocks, dtype=bool)

    # 截面 rank-percentile (v8.0: rank-product, 对齐 V2/V3 聚宽策略)
    r_arrays = {}
    for fname in g.weights.keys():
        arr = factor_arrays[fname]
        valid = factor_valid[fname]
        n_v = int(np.sum(valid))
        if n_v < 10:
            r_arrays[fname] = np.full(n_stocks, np.nan)
            continue
        # rank(pct=True) → [0, 1]
        ranks = np.full(n_stocks, np.nan)
        valid_vals = arr[valid]
        sorted_idx = np.argsort(valid_vals)
        ranks[valid] = np.interp(
            np.arange(n_v), [0, n_v - 1],
            [1.0 / n_v, 1.0]
        )[np.argsort(sorted_idx)]
        r_arrays[fname] = ranks

    # 加权秩乘积 (NaN-safe)
    log_composite = np.zeros(n_stocks)
    wsum = np.zeros(n_stocks)
    for fname, w in g.weights.items():
        r = r_arrays[fname]
        mask = np.isfinite(r) & (r > 1e-10)
        log_composite[mask] += abs(w) * np.log(r[mask])
        wsum[mask] += abs(w)

    mask_valid = wsum > 0
    composite = np.full(n_stocks, -999.0)
    composite[mask_valid] = np.exp(log_composite[mask_valid] / wsum[mask_valid])
    # Final rank percentile (截面排名)
    valid_comp = composite[mask_valid]
    n_vc = int(np.sum(mask_valid))
    if n_vc >= 10:
        sorted_idx = np.argsort(valid_comp)
        ranks_final = np.full(n_vc, np.nan)
        ranks_final[np.argsort(sorted_idx)] = np.linspace(1.0/n_vc, 1.0, n_vc)
        composite[mask_valid] = ranks_final
    n_composite = n_vc

    if g.debug:
        order = np.argsort(composite)[::-1]
        top_idx = order[:min(5, n_composite)]
        log.info('[%s] STEP4 composite: total=%d valid=%d top5_val=%s' % (
            date_str, n_stocks, n_composite,
            str(np.round(composite[top_idx][:5], 4))))

    if n_composite < g.stock_num:
        return

    sorted_idx = np.argsort(composite)[::-1]
    cand_n = min(len(sorted_idx), g.stock_num * 3)
    cand_codes = [stocks[i] for i in sorted_idx[:cand_n] if composite[i] > -900]
    try:
        cur = get_current_data()
    except Exception:
        cur = None
    target = []
    n_skip_pause = 0
    n_skip_limit = 0
    for s in cand_codes:
        if len(target) >= g.stock_num:
            break
        if cur is not None:
            try:
                cd = cur[s]
                if cd.paused:
                    n_skip_pause += 1
                    continue
                lp = getattr(cd, 'last_price', None)
                hl = getattr(cd, 'high_limit', None)
                if lp is not None and hl is not None and hl > 0 and lp >= hl - 1e-6:
                    n_skip_limit += 1
                    continue
            except Exception:
                pass
        target.append(s)

    if g.debug:
        log.info('[%s] STEP5 target=%d, skip_pause=%d skip_limit=%d, top3: %s' %
                 (date_str, len(target), n_skip_pause, n_skip_limit, str(target[:3])))

    rebalance(context, target)

'''

    def _build_stock_pool(self):
        return f'''
# ==================== 股票池 ====================
def get_stock_pool(context):
    prev = context.previous_date
    date_str_pool = prev.strftime('%Y-%m-%d')

    try:
        all_sec = get_all_securities(['stock'], date=date_str_pool)
        all_stocks = list(all_sec.index)
        if len(all_stocks) == 0:
            raise ValueError('empty')
    except:
        all_sec = get_all_securities(['stock'])
        all_stocks = list(all_sec.index)

    stocks = [s for s in all_stocks if not s.startswith(EXCLUDE_PREFIX)]

    try:
        current_data = get_current_data()
    except:
        current_data = {{}}

    filtered = []
    for s in stocks:
        try:
            if s in current_data:
                cd = current_data[s]
                if cd.is_st or (cd.name and 'ST' in str(cd.name)):
                    continue
                if cd.paused:
                    continue
        except:
            pass
        try:
            start_date = all_sec.loc[s, 'start_date']
            if start_date is not None:
                if not isinstance(start_date, pd.Timestamp):
                    start_date = pd.Timestamp(start_date)
                if (context.current_dt - start_date).days < 180:
                    continue
        except:
            pass
        filtered.append(s)

    if len(filtered) == 0:
        return []

    date_str = prev.strftime('%Y-%m-%d')
    mcap_ok = []
    for i in range(0, len(filtered), 2000):
        batch = filtered[i:i + 2000]
        try:
            q = query(
                valuation.code,
                valuation.circulating_market_cap
            ).filter(valuation.code.in_(batch))
            df = get_fundamentals(q, date=date_str)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    code = row['code']
                    mcap_val = row.get('circulating_market_cap')
                    if mcap_val is not None and not np.isnan(mcap_val) and mcap_val >= MCAP_MIN:
                        mcap_ok.append(code)
        except:
            pass

    if len(mcap_ok) > 1500:
        mcap_ok = mcap_ok[:1500]
    return mcap_ok

'''

    def _build_load_price_data(self):
        price_fields = get_required_price_fields(self.weights.keys())
        fields_py = str(price_fields)
        max_window = get_max_window(self.weights.keys()) + 5
        return f'''
# ==================== 数据加载 ====================
def load_price_data(stocks, context):
    n = len(stocks)
    end_dt = context.previous_date
    fields = {fields_py}
    result = {{f: {{}} for f in fields}}

    for field in fields:
        is_vol = field in ('volume',)
        for b in range(0, n, 500):
            batch = stocks[b:b + 500]
            try:
                px = get_price(batch, end_date=end_dt, count={max_window},
                               fields=[field], skip_paused=False)
                if px is None or len(px) == 0:
                    continue
                if isinstance(px, dict):
                    for s in batch:
                        if s in px and px[s] is not None and len(px[s]) > 0:
                            dfc = px[s]
                            col = field if field in dfc.columns else dfc.columns[0]
                            ser = dfc[col].ffill() if not is_vol else dfc[col].fillna(0)
                            result[field][s] = np.array(ser.values, dtype=float)
                else:
                    df = px[field] if hasattr(px, 'minor_axis') else px
                    df = df.ffill() if not is_vol else df.fillna(0)
                    for s in batch:
                        if s in df.columns:
                            result[field][s] = np.array(df[s].values, dtype=float)
            except Exception as e:
                if g.debug:
                    log.error('[load] %s FAILED: %s' % (field, str(e)))
    return result

'''

    def _build_factor_dispatch(self):
        """生成 compute_factor 调度函数"""
        cases = []
        for idx, fname in enumerate(self.weights.keys()):
            kw = "if" if idx == 0 else "elif"
            meta = JQ_FACTOR_REGISTRY[fname]
            # 基本面因子需要 context, 量价因子只需 price_data
            if meta['jq_data'] in ('fundamental', 'fundamental_multi'):
                call = f"compute_{fname}(stocks, context)"
            else:
                call = f"compute_{fname}(stocks, price_data)"
            cases.append(f"    {kw} fname == '{fname}':\n        return {call}")

        cases_str = "\n".join(cases)

        return f'''
# ==================== 因子调度 ====================
def compute_factor(fname, stocks, context, price_data):
    n = len(stocks)
{cases_str}
    else:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)

'''

    def _build_factor_functions(self):
        """生成所有因子的 compute 函数"""
        blocks = []
        for fname in self.weights.keys():
            meta = JQ_FACTOR_REGISTRY[fname]
            func_code = meta['jq_compute_function'].strip()
            blocks.append(func_code)
        return "\n\n".join(blocks) + "\n"

    def _build_utils(self):
        return '''
# ==================== 工具函数 ====================
def rank_standardize(arr, valid_mask):
    """截面 rank-based 标准化: percentile → [-3, 3] 线性映射

    免疫复权常数倍/价格缩放差异; 无需假定分布; 无clip敏感度.
    等价于 z-score 在均匀分布上的特例, 多因子行业标准.

    Args:
        arr: 原始因子值 array(N,)
        valid_mask: bool array(N,), True=有效
    Returns:
        array(N,) 标准化后因子值, 无效位置=-999.0
    """
    n = len(arr)
    result = np.full(n, -999.0)
    V = int(np.sum(valid_mask))
    if V < 10:
        return result
    vals = arr[valid_mask]
    order = np.argsort(vals)
    ranks = np.empty(V)
    ranks[order] = np.arange(1, V + 1)
    pct = ranks / (V + 1)  # (0, 1), 避免 0/1 极端
    z = (pct - 0.5) * 6   # [-3, 3] 均匀映射
    result[valid_mask] = z
    return result


def _isnan(x):
    try:
        return np.isnan(x)
    except:
        return False


# ==================== #12 补丁: 科创板/创业板保护限价 ====================
def board_limit_factors(code):
    if code.startswith(('688', '689')):
        return 1.2, 0.8
    if code.startswith(('300', '301')):
        return 1.2, 0.8
    if code.startswith(('8', '4')):
        return 1.3, 0.7
    return 1.1, 0.9


def order_target_with_limit(context, code, target_value, is_buy, prev_close=None):
    if target_value == 0:
        return order_target_value(code, 0)
    if not USE_PROTECTION_LIMIT:
        return order_target_value(code, target_value)
    limit_px = None
    if prev_close is not None and not (isinstance(prev_close, float) and np.isnan(prev_close)) \
            and prev_close > 0:
        up, down = board_limit_factors(code)
        limit_px = prev_close * (up if is_buy else down)
    if limit_px is not None:
        try:
            return order_target_value(code, target_value, LimitOrderStyle(limit_px))
        except Exception:
            pass
    return order_target_value(code, target_value)

'''

    def _build_rebalance(self):
        return '''
# ==================== 调仓 ====================
def rebalance(context, target):
    current_holdings = set(context.portfolio.positions.keys())
    target_set = set(target)
    to_sell = current_holdings - target_set

    all_codes = list(current_holdings | target_set)
    prev_close = {}
    if all_codes:
        try:
            pc = history(1, '1d', 'close', all_codes, skip_paused=False, df=False)
            for c in all_codes:
                if c in pc and len(pc[c]) > 0:
                    prev_close[c] = pc[c][-1]
        except Exception:
            prev_close = {}

    for s in to_sell:
        order_target_with_limit(context, s, 0, False, prev_close.get(s))

    n = len(target)
    if n == 0:
        return

    total_value = context.portfolio.total_value
    per_value = total_value / max(n, 1)

    turnover_rate = len(to_sell) / max(len(current_holdings), 1)
    if turnover_rate > g.max_turnover:
        max_new = int(n * g.max_turnover)
        target = target[:max_new]
        per_value = total_value / max(len(target), 1)

    for s in target:
        order_target_with_limit(context, s, per_value, True, prev_close.get(s))

    n_pos = len(context.portfolio.positions)
    invested = 0.0
    for p in context.portfolio.positions.values():
        try:
            invested += p.value
        except Exception:
            pass
    if g.debug:
        log.info('[FILLS] %s positions=%d invested=%.0f (%.1f%% of %.0f)' % (
            context.current_dt.strftime('%Y-%m-%d'), n_pos, invested,
            100.0 * invested / max(total_value, 1), total_value))

    g.last_holdings = target_set
'''
