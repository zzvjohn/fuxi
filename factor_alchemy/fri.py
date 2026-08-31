"""
FRI (Factor Robustness Index) — 因子稳健性指数

替代 jq_signal_preservation。纯统计质量评估，不依赖历史JQ结果。
避免"奖励长得像老赢家"的多样性灭绝问题。

四维度:
  precision:   Bootstrap ICIR 精度 (CI宽度→信号估计可靠性)
  persistence: 截面排序持续性 (Spearman ρ周间相关→信号结构真实性)
  consistency: Regime 一致性 (跨环境ICIR正向→稳健性)
  novelty:     因子新颖性 (1 - max_corr_with_existing→多样性保护)

FRI = 0.25 × precision + 0.25 × persistence + 0.25 × consistency + 0.25 × novelty

用法:
    from research.factor_alchemy.fri import compute_fri, FRIResult
    result = compute_fri(
        factor_name="earnings_pre_drift_alignment",
        factor_values=pd.DataFrame(...),  # 截面: date × stock
        forward_returns=pd.DataFrame(...),
        regime_labels=pd.Series(...),
        existing_factor_pool={...},  # 已有因子名称→因子值
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

# ── 常数 ──────────────────────────────────────────────────
BOOTSTRAP_B = 500       # Bootstrap 重采样次数
BOOTSTRAP_BLOCK = 5     # 块长 (天), 保留自相关结构
MIN_IC_POINTS = 20      # 最少IC点数才计算ICIR
REGIME_MIN_WEEKS = 10   # 每个regime最少周数才纳入consistency计算
PERSISTENCE_WEEKS = 4   # 排序持续性评估窗口(周)


@dataclass
class FRIResult:
    """FRI 计算结果"""
    factor_name: str
    precision: float = 0.0       # [0,1]
    persistence: float = 0.0     # [0,1]
    consistency: float = 0.0     # [0,1]
    novelty: float = 0.0         # [0,1]
    fri: float = 0.0             # [0,1]
    # 诊断信息
    icir_point: float = np.nan
    icir_ci_low: float = np.nan
    icir_ci_high: float = np.nan
    weekly_rho_mean: float = np.nan
    regime_icirs: Dict[str, float] = field(default_factory=dict)
    max_corr_with_existing: float = 0.0
    max_corr_factor: str = ""
    warnings: List[str] = field(default_factory=list)
    grade: str = ""              # A(>=0.60) / B(0.40-0.60) / C(0.25-0.40) / D(<0.25)


def compute_fri(
    factor_name: str,
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    existing_factor_pool: Optional[Dict[str, pd.DataFrame]] = None,
    ic_series: Optional[pd.Series] = None,
    verbose: bool = False,
) -> FRIResult:
    """
    计算单个因子的 FRI。

    Parameters
    ----------
    factor_name: 因子名称
    factor_values: 截面因子值, index=date, columns=stock_code
    forward_returns: 前向收益, 格式同 factor_values
    regime_labels: 日期→regime标签 (0=small_bull, 1=large_dominant)
    existing_factor_pool: 已有因子 {name: DataFrame(factor_values)}
    ic_series: 预计算的IC序列 (可复用外部计算结果)
    verbose: 是否打印诊断信息

    Returns
    -------
    FRIResult dataclass
    """
    result = FRIResult(factor_name=factor_name)
    
    # ── 0. 预计算 IC 序列 ──────────────────────────────────
    if ic_series is None:
        ic_series = _compute_ic_series(factor_values, forward_returns)
    if ic_series is None or len(ic_series) < MIN_IC_POINTS:
        result.warnings.append(f"IC点数不足 ({len(ic_series) if ic_series is not None else 0} < {MIN_IC_POINTS})")
        return result
    
    # ── 1. precision: Bootstrap ICIR 精度 ──────────────────
    precision, icir_pt, ci_low, ci_high = _compute_precision(ic_series)
    result.precision = precision
    result.icir_point = icir_pt
    result.icir_ci_low = ci_low
    result.icir_ci_high = ci_high
    
    # ── 2. persistence: 截面排序持续性 ─────────────────────
    persistence, weekly_rho = _compute_persistence(factor_values)
    result.persistence = persistence
    result.weekly_rho_mean = weekly_rho
    
    # ── 3. consistency: Regime 一致性 ──────────────────────
    if regime_labels is not None and len(regime_labels) > 0:
        consistency, regime_icirs = _compute_consistency(ic_series, regime_labels)
        result.consistency = consistency
        result.regime_icirs = regime_icirs
    else:
        result.consistency = 0.5  # 无regime数据时中性
        if verbose:
            result.warnings.append("无regime数据, consistency=0.5(中性)")
    
    # ── 4. novelty: 因子新颖性 ─────────────────────────────
    if existing_factor_pool is not None and len(existing_factor_pool) > 0:
        novelty, max_corr, max_name = _compute_novelty(
            factor_values, existing_factor_pool, factor_name
        )
        result.novelty = novelty
        result.max_corr_with_existing = max_corr
        result.max_corr_factor = max_name
    else:
        result.novelty = 1.0  # 无已有因子 → 完全新颖
        if verbose:
            result.warnings.append("无已有因子池, novelty=1.0(默认)")
    
    # ── 汇总 FRI ──────────────────────────────────────────
    result.fri = round(0.25 * result.precision + 0.25 * result.persistence
                        + 0.25 * result.consistency + 0.25 * result.novelty, 4)
    
    # ── 等级 ───────────────────────────────────────────────
    if result.fri >= 0.60:
        result.grade = "A"
    elif result.fri >= 0.40:
        result.grade = "B"
    elif result.fri >= 0.25:
        result.grade = "C"
    else:
        result.grade = "D"
    
    if verbose:
        print(f"[FRI] {factor_name}: {result.fri:.3f} ({result.grade}) "
              f"| P={result.precision:.3f} S={result.persistence:.3f} "
              f"C={result.consistency:.3f} N={result.novelty:.3f}")
    
    return result


def compute_fri_batch(
    factors: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    existing_pool: Optional[Dict[str, pd.DataFrame]] = None,
    ic_series_cache: Optional[Dict[str, pd.Series]] = None,
    verbose: bool = False,
) -> Dict[str, FRIResult]:
    """
    批量计算多个因子的 FRI。

    Parameters
    ----------
    factors: {name: factor_values_df}
    forward_returns: 前向收益
    regime_labels: regime标签
    existing_pool: 已有因子池(用于novelty计算, 不包含factors中的因子)
    ic_series_cache: 预计算的IC序列缓存
    verbose: 详细输出
    """
    results = {}
    for name, fv in factors.items():
        # 构建"已有因子池"：排除当前正在计算的因子
        pool = {}
        if existing_pool:
            pool.update(existing_pool)
        # 加入同批次中已计算的因子（用于新颖性）
        for n2, r2 in results.items():
            if n2 != name:
                # 这里需要缓存因子值——简化：用existing_pool + other_results的因子值
                pass
        
        ic = ic_series_cache.get(name) if ic_series_cache else None
        results[name] = compute_fri(
            factor_name=name,
            factor_values=fv,
            forward_returns=forward_returns,
            regime_labels=regime_labels,
            existing_factor_pool=pool,
            ic_series=ic,
            verbose=verbose,
        )
    return results


# ═══════════════════════════════════════════════════════════
# 子函数
# ═══════════════════════════════════════════════════════════

def _compute_ic_series(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_stocks: int = 30,
) -> Optional[pd.Series]:
    """计算日频Rank IC序列"""
    ic_vals = {}
    common_dates = factor_values.index.intersection(forward_returns.index)
    
    for dt in sorted(common_dates):
        fv = factor_values.loc[dt].dropna()
        fr = forward_returns.loc[dt].dropna()
        common = fv.index.intersection(fr.index)
        if len(common) < min_stocks:
            continue
        ic = pd.Series(fv[common]).rank().corr(pd.Series(fr[common]).rank())
        if not np.isnan(ic):
            ic_vals[dt] = ic
    
    if len(ic_vals) < MIN_IC_POINTS:
        return None
    
    return pd.Series(ic_vals).sort_index()


def _compute_precision(ic_series: pd.Series) -> Tuple[float, float, float, float]:
    """
    Block bootstrap → ICIR经验分布 → CI宽度 → precision

    Returns: (precision, icir_point_estimate, ci_lower, ci_upper)
    """
    n = len(ic_series)
    icir_boot = []
    rng = np.random.RandomState(42)
    
    for _ in range(BOOTSTRAP_B):
        # Block bootstrap: 保留自相关结构
        n_blocks = max(1, n // BOOTSTRAP_BLOCK)
        starts = rng.randint(0, max(1, n - BOOTSTRAP_BLOCK + 1), size=n_blocks)
        indices = []
        for s in starts:
            indices.extend(range(s, min(s + BOOTSTRAP_BLOCK, n)))
        indices = indices[:n]  # 截断到n
        
        boot_sample = ic_series.iloc[indices]
        mu = boot_sample.mean()
        sigma = boot_sample.std()
        if sigma > 1e-10:
            icir_boot.append(abs(mu / sigma))
    
    if len(icir_boot) < 50:
        return 0.0, np.nan, np.nan, np.nan
    
    icir_boot = np.array(icir_boot)
    ci_low = np.percentile(icir_boot, 2.5)
    ci_high = np.percentile(icir_boot, 97.5)
    ci_width = ci_high - ci_low
    
    # 点估计
    mu = ic_series.mean()
    sigma = ic_series.std()
    icir_pt = abs(mu / sigma) if sigma > 1e-10 else 0.0
    
    # precision: CI越窄→精度越高
    if ci_width < 1e-10:
        precision = 1.0
    else:
        precision = min(1.0 / ci_width, 1.0)
    
    return round(precision, 4), round(icir_pt, 4), round(ci_low, 4), round(ci_high, 4)


def _compute_persistence(factor_values: pd.DataFrame) -> Tuple[float, float]:
    """
    计算相邻周之间的截面排序 Spearman ρ。

    取最近4周，每相邻两周计算截面rank correlation，取均值。

    Returns: (persistence [0,1], mean_weekly_rho)
    """
    if factor_values.empty or len(factor_values) < 10:
        return 0.0, np.nan
    
    # 降采样到周频 (取每周最后一个交易日)
    weekly = factor_values.resample('W').last().dropna(how='all')
    
    if len(weekly) < 3:
        # 不足3周，降级到双周
        weekly = factor_values.resample('2W').last().dropna(how='all')
    
    if len(weekly) < 2:
        return 0.0, np.nan
    
    # 取最近 PERSISTENCE_WEEKS 周
    weekly = weekly.iloc[-PERSISTENCE_WEEKS:]
    
    rho_vals = []
    for i in range(len(weekly) - 1):
        w1 = weekly.iloc[i].dropna()
        w2 = weekly.iloc[i + 1].dropna()
        common = w1.index.intersection(w2.index)
        if len(common) < 20:
            continue
        rho = w1[common].corr(w2[common], method='spearman')
        if not np.isnan(rho):
            rho_vals.append(rho)
    
    if not rho_vals:
        return 0.0, np.nan
    
    mean_rho = np.mean(rho_vals)
    # 取绝对值后截断到 [0, 1]
    persistence = min(abs(mean_rho), 1.0)
    
    return round(persistence, 4), round(mean_rho, 4)


def _compute_consistency(
    ic_series: pd.Series,
    regime_labels: pd.Series,
) -> Tuple[float, Dict[str, float]]:
    """
    计算 ICIR 在三个 regime (small_bull/large_dominant/neutral) 中的一致性。

    consistency = Σ (ICIR_regime > 0 ? 1/3 : 0)
    如果有 regime 的 ICIR < -0.10 → 额外惩罚 -0.15

    参数:
      ic_series: index=date, values=IC
      regime_labels: index=date, values=0(small_bull) or 1(large_dominant)
    """
    # regime 编码
    REGIME_MAP = {0: "small_bull", 1: "large_dominant"}
    
    # 将 regime_labels 映射到每个IC日期
    # regime_labels 可能是日频或周频, 需要对齐
    regime_icirs = {}
    consistency = 0.0
    
    # 尝试对齐: 对于每个 regime 类别
    for regime_code, regime_name in REGIME_MAP.items():
        # 找到属于该 regime 的日期
        regime_dates = regime_labels[regime_labels == regime_code].index
        
        if len(regime_dates) == 0:
            continue
        
        # 取IC序列中属于该regime的部分
        ic_sub = ic_series[ic_series.index.isin(regime_dates)]
        
        # 如果IC是日频, regime是周频, 需要resample对齐
        if len(ic_sub) < REGIME_MIN_WEEKS:
            # 尝试宽松匹配: 将IC日期映射到最近的regime日期
            ic_sub = pd.Series(dtype=float)
            for dt in ic_series.index:
                # 找到最近的 regime 日期
                diffs = abs((regime_dates - dt).days if hasattr(regime_dates - dt, 'days') 
                           else abs(pd.Timestamp(rd) - pd.Timestamp(dt)) for rd in regime_dates)
                min_diff = min(diffs) if diffs else float('inf')
                if min_diff <= 3:  # 3天内
                    ic_sub[dt] = ic_series[dt]
        
        if len(ic_sub) < REGIME_MIN_WEEKS:
            continue
        
        mu = ic_sub.mean()
        sigma = ic_sub.std()
        icir = abs(mu / sigma) if sigma > 1e-10 else 0.0
        
        # 符号保持: 记录有符号的ICIR
        regime_icirs[regime_name] = round(mu / sigma, 4) if sigma > 1e-10 else 0.0
        
        if mu > 0:
            consistency += 1.0 / len(REGIME_MAP)
        elif mu / sigma > -0.10 if sigma > 1e-10 else True:
            # ICIR > -0.10: 不算正向但也无害 → 不计入但不惩罚
            pass
    
    # 惩罚: 任何 regime 的 ICIR < -0.10
    for rname, ricir in regime_icirs.items():
        if ricir < -0.10:
            consistency -= 0.15
    
    # 如果没有有效regime数据, 返回中性
    if not regime_icirs:
        consistency = 0.5
    
    consistency = max(0.0, min(1.0, consistency))
    
    return round(consistency, 4), regime_icirs


def _compute_novelty(
    factor_values: pd.DataFrame,
    existing_pool: Dict[str, pd.DataFrame],
    factor_name: str,
) -> Tuple[float, float, str]:
    """
    计算因子新颖性: 1 - max(截面Spearman相关)

    新颖性越高, 因子与已有因子池的正交性越好。

    Returns: (novelty [0,1], max_correlation, most_correlated_factor_name)
    """
    if not existing_pool:
        return 1.0, 0.0, ""
    
    max_corr = 0.0
    max_name = ""
    
    # 取最近一个共同日期做截面相关
    latest_date = factor_values.dropna(how='all').index[-1]
    fv_latest = factor_values.loc[latest_date].dropna()
    
    for ename, evalues in existing_pool.items():
        if ename == factor_name:
            continue
        try:
            if latest_date not in evalues.index:
                continue
            ev_latest = evalues.loc[latest_date].dropna()
            common = fv_latest.index.intersection(ev_latest.index)
            if len(common) < 20:
                continue
            corr = abs(fv_latest[common].corr(ev_latest[common], method='spearman'))
            if not np.isnan(corr) and corr > max_corr:
                max_corr = corr
                max_name = ename
        except Exception:
            continue
    
    novelty = 1.0 - max_corr
    return round(novelty, 4), round(max_corr, 4), max_name


# ═══════════════════════════════════════════════════════════
# 便捷函数: 从 Stage 2 输出计算 FRI
# ═══════════════════════════════════════════════════════════

def _load_extra_data(data_dir: Path, dates_index, stocks_columns) -> dict:
    """
    P-018/P-008/P-009: 加载 moneyflow + balancesheet 额外数据，
    转换为与 OHLCV 对齐的 DataFrame 格式 (date × stock)。
    仅在公式中引用这些变量时才需要。
    """
    extra = {}
    try:
        # moneyflow_daily → buy_lg_vol, sell_lg_vol, buy_sm_vol, sell_sm_vol, buy_md_vol, sell_md_vol
        mf_path = data_dir / "raw" / "moneyflow_daily.csv"
        if mf_path.exists():
            mf = pd.read_csv(mf_path, dtype={'trade_date': str})
            mf['trade_date'] = pd.to_datetime(mf['trade_date'], format='%Y%m%d')
            flow_cols = ['buy_lg_vol','sell_lg_vol','buy_sm_vol','sell_sm_vol',
                        'buy_md_vol','sell_md_vol','buy_elg_vol','sell_elg_vol']
            for col in flow_cols:
                if col in mf.columns:
                    piv = mf.pivot_table(values=col, index='trade_date', columns='ts_code', aggfunc='sum')
                    # 对齐 OHLCV 数据
                    aligned = pd.DataFrame(index=dates_index, columns=stocks_columns, dtype=float)
                    common_idx = piv.index.intersection(dates_index)
                    common_cols = piv.columns.intersection(stocks_columns)
                    if len(common_idx) > 0 and len(common_cols) > 0:
                        aligned.loc[common_idx, common_cols] = piv.loc[common_idx, common_cols].values
                    extra[col] = aligned

        # balancesheet → intan_assets, goodwill, total_assets
        bs_path = data_dir / "raw" / "balancesheet.csv"
        if bs_path.exists():
            bs = pd.read_csv(bs_path, dtype={'end_date': str})
            bs['end_date'] = pd.to_datetime(bs['end_date'])
            bs_cols = ['total_assets','intan_assets','goodwill']
            for col in bs_cols:
                if col in bs.columns:
                    piv = bs.pivot_table(values=col, index='end_date', columns='ts_code', aggfunc='first')
                    aligned = pd.DataFrame(index=dates_index, columns=stocks_columns, dtype=float)
                    common_idx = piv.index.intersection(dates_index)
                    common_cols = piv.columns.intersection(stocks_columns)
                    if len(common_idx) > 0 and len(common_cols) > 0:
                        aligned.loc[common_idx, common_cols] = piv.loc[common_idx, common_cols].values
                    extra[col] = aligned
    except Exception as e:
        print(f"[FRI] 额外数据加载失败 (非致命): {e}")
    return extra


def compute_fri_from_stage2(
    daily_json_path: Path,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    existing_pool: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, FRIResult]:
    """
    从 Stage 2 的 daily_factor_YYYYMMDD.json 读取结果,
    计算每个因子的 FRI。

    JSON 中的 results 数组应包含: name, formula, +ic% 等字段。
    我们重新执行公式计算因子值, 然后计算 FRI。
    """
    import json
    
    with open(daily_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        return {}
    
    # 前向收益: 5日
    from datetime import timedelta
    forward_5d = close_df.shift(-5) / close_df - 1

    # P-018: 加载 moneyflow/balancesheet 额外数据
    from pathlib import Path
    data_dir = daily_json_path.parent.parent / "raw" if "raw" not in str(daily_json_path) else daily_json_path.parent
    if not data_dir.exists():
        data_dir = Path(daily_json_path).resolve().parent.parent / "data" / "raw"
    extra_data = _load_extra_data(data_dir, close_df.index, close_df.columns)

    fri_results = {}
    for r in results:
        name = r.get('name', '')
        formula = r.get('formula', '')
        direction = r.get('direction', 'long')
        
        if not formula:
            continue
        
        # 执行公式计算因子值
        try:
            factor_df = _exec_formula(formula, close_df, open_df, high_df, low_df, vol_df, extra_data)
            if factor_df is None or factor_df.empty:
                continue
            
            # 方向调整
            if direction == 'short':
                factor_df = -factor_df
            
            # 截面标准化
            factor_df = _cross_section_zscore(factor_df)
            
            # 计算 FRI
            fr = compute_fri(
                factor_name=name,
                factor_values=factor_df,
                forward_returns=forward_5d,
                regime_labels=regime_labels,
                existing_factor_pool=existing_pool,
                verbose=True,
            )
            fri_results[name] = fr
        except Exception as e:
            print(f"[FRI] 因子 '{name}' 计算失败: {e}")
    
    return fri_results


def _exec_formula(
    formula: str,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    extra_data: dict = None,
) -> Optional[pd.DataFrame]:
    """执行因子公式, 返回因子值 DataFrame"""
    # 构建安全执行环境
    exec_globals = {
        'np': np,
        'pd': pd,
        'close_p': close_df,
        'open_p': open_df,
        'high_p': high_df,
        'low_p': low_df,
        'vol_p': vol_df,
        'volume_p': vol_df,   # 兼容 daily_factor_hypothesis.py 的变量名
        'volume': vol_df,
        'close': close_df,
        'open': open_df,
        'high': high_df,
        'low': low_df,
    }

    # P-018/P-008/P-009: 注入额外数据列 (moneyflow/balancesheet)
    if extra_data:
        for col_name, col_df in extra_data.items():
            if isinstance(col_df, pd.DataFrame):
                exec_globals[col_name] = col_df
    exec_locals = {}
    
    try:
        # 尝试单行表达式
        result = eval(formula, exec_globals, exec_locals)
        if isinstance(result, pd.DataFrame):
            return result
    except Exception:
        pass
    
    try:
        # 多行代码块
        formula = formula.strip()
        if 'import' in formula[:50]:
            # 包含 import 语句的完整代码块
            lines = [l for l in formula.split('\n') if l.strip()]
            import_lines = []
            code_lines = []
            in_imports = True
            for line in lines:
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_lines.append(line)
                else:
                    in_imports = False
                    code_lines.append(line)
            
            # 执行 imports
            for il in import_lines:
                exec(il, exec_globals)
            
            # 执行主体代码
            code = '\n'.join(code_lines)
            exec(code, exec_globals, exec_locals)
        else:
            exec(formula, exec_globals, exec_locals)
        
        # 查找 result 变量
        for key in ['result', 'factor_values', 'fv', 'factor']:
            if key in exec_locals:
                val = exec_locals[key]
                if isinstance(val, pd.DataFrame):
                    return val
                elif isinstance(val, np.ndarray) and val.ndim == 2:
                    return pd.DataFrame(val, index=close_df.index, columns=close_df.columns)
        
        # 最后一个赋值变量
        if exec_locals:
            last_val = list(exec_locals.values())[-1]
            if isinstance(last_val, pd.DataFrame):
                return last_val
    except Exception as e:
        print(f"[FRI] 公式执行失败: {e}")
    
    return None


def _cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score 标准化 + 缩尾"""
    result = df.copy()
    for idx in range(len(df)):
        row = df.iloc[idx].dropna()
        if len(row) < 20:
            continue
        # 缩尾 1%/99%
        lo = row.quantile(0.01)
        hi = row.quantile(0.99)
        row = row.clip(lo, hi)
        # z-score
        mu = row.mean()
        sigma = row.std()
        if sigma > 1e-10:
            row = (row - mu) / sigma
        result.iloc[idx, :len(row)] = row.values
    return result


def build_existing_pool_from_csv(
    pool_csv_path: Path,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    vol_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    从 passed_factor_pool.csv 构建已有因子池 (用于 novelty 计算)。

    仅处理 status=candidate (PASS) 和 reserve 且有 ICIR 的因子。
    """
    if not pool_csv_path.exists():
        return {}
    
    pool_df = pd.read_csv(pool_csv_path, encoding='utf-8-sig')
    
    # 过滤: 有公式且有ICIR的因子
    pool_df = pool_df[pool_df['formula'].notna() & (pool_df['formula'] != '')]
    
    # 加载额外数据
    from pathlib import Path as _Path
    data_dir = _Path(pool_csv_path).parent.parent / "raw" if "raw" not in str(pool_csv_path) else _Path(pool_csv_path).parent
    extra_data = _load_extra_data(data_dir, close_df.index, close_df.columns)

    existing = {}
    for _, row in pool_df.iterrows():
        name = row['name']
        formula = row['formula']
        try:
            fv = _exec_formula(formula, close_df, open_df, high_df, low_df, vol_df, extra_data)
            if fv is not None and not fv.empty:
                existing[name] = _cross_section_zscore(fv)
        except Exception:
            continue
    
    return existing


# ═══════════════════════════════════════════════════════════
# 适应度计算 (整合 FRI)
# ═══════════════════════════════════════════════════════════

def compute_evolution_fitness(
    icir_consistency: float,      # 多窗口ICIR一致性 [0,1]
    fri_result: FRIResult,        # FRI结果
    n_factors: int = 4,           # 策略中使用的因子数
    n_pairs: int = 2,             # 配对数
    composite_depth: int = 1,     # 嵌套深度
) -> Dict[str, float]:
    """
    进化适应度计算 (替代旧 GeneFitness 的 local_sharpe 驱动)

    F = 0.45 × icir_consistency + 0.40 × FRI - 0.15 × complexity_penalty
    """
    # 复杂度惩罚
    cp_signal = max(0, n_factors - 4) / 10.0
    cp_pair = max(0, n_pairs - 2) / 8.0
    cp_depth = max(0, composite_depth - 1) / 6.0
    complexity_penalty = min(1.0, cp_signal + cp_pair + cp_depth)
    
    fitness = (
        0.45 * icir_consistency
        + 0.40 * fri_result.fri
        - 0.15 * complexity_penalty
    )
    
    return {
        'fitness': round(max(0.0, fitness), 4),
        'icir_consistency': icir_consistency,
        'fri': fri_result.fri,
        'fri_grade': fri_result.grade,
        'complexity_penalty': round(complexity_penalty, 4),
        'n_factors': n_factors,
        'n_pairs': n_pairs,
    }


def compute_icir_consistency(
    ic_series: pd.Series,
    windows: List[int] = [52, 26, 13, 4],
) -> float:
    """
    计算多窗口 ICIR 一致性。

    对每个窗口 (单位: 周), 计算最近N周的ICIR,
    然后评估跨窗口稳定性。

    Returns: icir_consistency ∈ [0, 1]
    """
    n = len(ic_series)
    if n < max(windows):
        return 0.0
    
    icirs = {}
    signs = []
    for w in windows:
        sub = ic_series.iloc[-w:] if n >= w else ic_series
        if len(sub) < 5:
            continue
        mu = sub.mean()
        sigma = sub.std()
        icir = abs(mu / sigma) if sigma > 1e-10 else 0.0
        icirs[w] = icir
        signs.append(1 if mu > 0 else 0)
    
    if not icirs:
        return 0.0
    
    icir_vals = list(icirs.values())
    mean_icir = np.mean(icir_vals)
    std_icir = np.std(icir_vals)
    
    # 稳定性: 1 - CV
    stability = 1.0 - (std_icir / mean_icir) if mean_icir > 1e-10 else 0.0
    stability = max(0.0, min(1.0, stability))
    
    # 方向一致性
    sign_concordance = sum(signs) / len(signs)  # 全正向=1, 全负向=0
    
    return round(mean_icir * stability * sign_concordance, 4)
