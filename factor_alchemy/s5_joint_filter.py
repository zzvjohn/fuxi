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
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


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
    
    # 计算状态
    error: str = ""
    computation_time: float = 0.0

    # v0.6.1: 组合收益序列 (内存对象, 不序列化; S6 DSR + 增量门禁的数据通道)
    portfolio_returns: Any = None
    portfolio_returns_n: int = 0
    
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
        sample_stocks: int = 200,
        rebalance_freq: str = "W",  # Weekly
        cache_dir: Path = None,
    ):
        self.price_path = price_path or RAW_DIR / "daily_prices.csv"
        self.index_path = index_path or RAW_DIR / "index_000905_daily.csv"
        self.mcap_path = mcap_path or DATA_DIR / "jqdata" / "mcap_weekly.csv"
        self.top_n = top_n
        self.excess_threshold = excess_threshold
        self.calmar_threshold = calmar_threshold
        self.sample_stocks = sample_stocks
        self.rebalance_freq = rebalance_freq
        self.cache_dir = cache_dir or DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 延迟加载
        self._price_df: Optional[pd.DataFrame] = None
        self._index_df: Optional[pd.DataFrame] = None
        self._universe: Optional[List[str]] = None
        self._loaded = False

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
        _read_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
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
        df = df[(df["trade_date"] >= "2020-01-01") & (df["trade_date"] <= "2026-08-28")]

        # v0.9.3: 数值列转 float32 (rank 相关/回测精度无影响, 内存减半)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        
        # 创建 _p 后缀字段 (pandas 公式使用的字段)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
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

        # 加载中证500指数
        print(f"  [S5] Loading index data from {self.index_path}...")
        idx_df = pd.read_csv(self.index_path)
        idx_df["trade_date"] = pd.to_datetime(
            idx_df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
            format="mixed", errors="coerce")
        idx_df = idx_df.dropna(subset=["trade_date"]).sort_values("trade_date")
        idx_df["idx_return"] = idx_df["close"].pct_change()
        self._index_df = idx_df
        print(f"    Loaded {len(idx_df)} index days")

        # 选择 universe (采样流动性最好的 stocks)
        self._select_universe()

        # v0.9.3 内存修复: universe 选定后立即裁剪 price_df 到 universe 股票,
        # 再 merge 辅助数据。旧实现先 merge 后裁剪 → 13.3M 行 × 25 列
        # merge_asof 时 pandas consolidate 需一次性 (25, 13303024) float64
        # ≈ 2.48 GiB 连续内存, Windows 下必爆 (_ArrayMemoryError 2026-08-21 实证)。
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
                    df = df.merge(mf, on=["ts_code", "trade_date"], how="left")
                    for col in avail_mf:
                        df[col] = df[col].fillna(0)
                        df[f"{col}_p"] = df[col]
                    mf_loaded = True
                    print(f"  [S5] Loaded moneyflow: {len(avail_mf)} fields ({', '.join(avail_mf)})")
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
                    c for c in ["rzye", "rqye", "rzmre", "rqyl", "rqchl"]
                    if c in mg.columns
                ]
                if len(mg_cols) > 2:
                    mg = mg[mg_cols].dropna(subset=["trade_date"])
                    df = df.merge(mg, on=["ts_code", "trade_date"], how="left")
                    for col in ["rzye", "rqye", "rzmre", "rqyl", "rqchl"]:
                        if col in df.columns:
                            df[f"{col}_p"] = df[col]
                    print(f"  [S5] Loaded margin: "
                          f"{len([c for c in ['rzye', 'rqye', 'rzmre', 'rqyl', 'rqchl'] if c in df.columns])} fields "
                          f"(至 {mg['trade_date'].max().date()})")
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
                    merged = pd.merge_asof(
                        df_sorted.sort_values("trade_date"),
                        bs_sorted.sort_values("trade_date"),
                        by="ts_code",
                        on="trade_date",
                        direction="backward",
                        tolerance=pd.Timedelta(days=180),  # 最多回溯半年
                    )
                    for col in avail_bs:
                        merged[col] = merged[col].fillna(0)
                        merged[f"{col}_p"] = merged[col]
                    df = merged
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
                df = df.merge(ba, on=["ts_code", "trade_date"], how="left")
                for col in _val_cols:
                    if col in df.columns:
                        df[f"{col}_p"] = df[col]
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
                merged = pd.merge_asof(
                    df_sorted.sort_values("trade_date"),
                    fi_sorted.sort_values("trade_date"),
                    by="ts_code", on="trade_date", direction="backward",
                    tolerance=pd.Timedelta(days=370),
                )
                for col in _fina_cols:
                    if col in merged.columns:
                        merged[f"{col}_p"] = merged[col]
                df = merged
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
                df = df.merge(agg, on=["ts_code", "trade_date"], how="left")
                for col in ["lhb_flag", "lhb_net_amount", "lhb_amount", "lhb_net_rate"]:
                    df[col] = df[col].fillna(0)
                    df[f"{col}_p"] = df[col]
                print(f"  [S5] Loaded top_list: 4 fields (lhb_flag, lhb_net_amount, "
                      f"lhb_amount, lhb_net_rate)")
            except Exception as e:
                print(f"  [S5] ⚠️ top_list load failed: {e}")
        else:
            print(f"  [S5] ⚠️ top_list.csv not found — lhb 类模板将失败")

        # v0.10: 慢变量按股 ffill (估值日缺失 1-2 天 / 财报公告间隔)
        _slow_cols = [c for c in ["pe_ttm", "pb", "dv_ratio",
                                  "roe_dt", "tr_yoy", "netprofit_yoy"] if c in df.columns]
        if _slow_cols:
            df = df.sort_values(["ts_code", "trade_date"])
            df[_slow_cols] = df.groupby("ts_code")[_slow_cols].ffill()

        self._price_df = df

    def _select_universe(self):
        """选择回测 universe: 按成交量排序取 top-N"""
        if self._price_df is None:
            self._load_data()
        
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

        df = self._price_df[self._price_df["ts_code"].isin(self._universe)].copy()
        
        # ── 公式预处理 ─────────────────────────────────────
        clean_formula = formula.strip()

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
            else:
                # ── 单股票模式: 逐股 eval ──
                return self._evaluate_per_stock(df, clean_formula, factor_name)
        except Exception as e:
            print(f"    [S5] Error evaluating {factor_name}: {e}")
            return None

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
            "rzye", "rqye", "rzmre", "rqyl", "rqchl",
            "rzye_p", "rqye_p", "rzmre_p", "rqyl_p", "rqchl_p",
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
        for pf in price_fields:
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
                          # v0.6.2 (2026-08-29): 两融字段
                          'rzye', 'rqye', 'rzmre', 'rqyl', 'rqchl',
                          'total_assets', 'intan_assets', 'goodwill',
                          # v0.10: 估值 + 财务质量字段
                          'pe_ttm', 'pb', 'dv_ratio', 'roe_dt', 'tr_yoy', 'netprofit_yoy',
                          # v0.6.1: 龙虎榜字段
                          'lhb_flag', 'lhb_net_amount', 'lhb_amount', 'lhb_net_rate']:
            if extra_col in df.columns:
                fields.append(extra_col)
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
                result_long = result_long.dropna(subset=['factor_value'])
                result_long = result_long[np.isfinite(result_long['factor_value'])]
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

    def backtest_factor(
        self,
        factor_df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        简单回测: 周频调仓，Top-N 等权做多。
        
        Returns:
          portfolio_returns: Series (trade_date index) 组合日收益率
          turnover_history: Series (trade_date index) 每次调仓的换手率
        """
        if factor_df is None or len(factor_df) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # 合并价格数据
        price = self._price_df[self._price_df["ts_code"].isin(self._universe)][
            ["ts_code", "trade_date", "close", "close_p"]
        ].copy()

        merged = factor_df.merge(price, on=["ts_code", "trade_date"], how="inner")
        if len(merged) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # 按周分组，取每周最后一个交易日
        merged["week"] = merged["trade_date"].dt.isocalendar().year.astype(str) + "-W" + \
                         merged["trade_date"].dt.isocalendar().week.astype(str).str.zfill(2)
        
        # 每周最后一天
        week_last = merged.groupby("week")["trade_date"].max().reset_index()
        week_last.columns = ["week", "rebalance_date"]
        
        rebalance_dates = sorted(week_last["rebalance_date"].unique())
        
        if len(rebalance_dates) < 2:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # 逐周调仓
        portfolio_returns = []
        
        for i in range(len(rebalance_dates) - 1):
            entry_date = rebalance_dates[i]
            exit_date = rebalance_dates[i + 1]
            
            # 选股: entry_date 的因子值排名
            entry_data = merged[merged["trade_date"] == entry_date].copy()
            if len(entry_data) < self.top_n:
                continue
            
            # 按因子值排序 (假设高值做多)
            entry_data = entry_data.sort_values("factor_value", ascending=False)
            selected = entry_data.head(self.top_n)
            selected_codes = set(selected["ts_code"].tolist())
            
            # 持仓期间收益
            period_data = merged[
                (merged["trade_date"] > entry_date) &
                (merged["trade_date"] <= exit_date) &
                (merged["ts_code"].isin(selected_codes))
            ].copy()
            
            if len(period_data) == 0:
                continue
            
            # 计算每日组合收益 (等权)
            period_data["daily_ret"] = period_data.groupby("ts_code")["close_p"].pct_change()
            daily_portfolio = period_data.groupby("trade_date")["daily_ret"].mean()
            
            portfolio_returns.append(daily_portfolio)

        if not portfolio_returns:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        port_ret = pd.concat(portfolio_returns).sort_index()
        # 去重 (同一日期取平均)
        port_ret = port_ret.groupby(port_ret.index).mean()
        
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

    def compute_metrics(
        self,
        portfolio_returns: pd.Series,
        year: int,
    ) -> Dict[str, float]:
        """
        计算给定年份的组合指标。
        
        Returns:
          {total_return, benchmark_return, excess_return, calmar, max_drawdown, annual_vol, periods_per_year}
        """
        if portfolio_returns.empty or self._index_df is None:
            return {"total_return": 0, "benchmark_return": 0, "excess_return": 0,
                    "calmar": 0, "max_drawdown": 0, "annual_vol": 0, "periods_per_year": 252}

        # v0.8 自动频次判断: 从完整收益序列推断年化因子 (防御周频/月频数据源)
        ppy = infer_periods_per_year(portfolio_returns.index)

        # 过滤年份
        port_year = portfolio_returns[
            (portfolio_returns.index >= f"{year}-01-01") &
            (portfolio_returns.index <= f"{year}-12-31")
        ].copy()

        # 最小观测守卫: 按频次自适应 (月频一年12观测不误杀; 日频保持>=50)
        min_obs = max(6, min(50, ppy // 2))
        if len(port_year) < min_obs:
            return {"total_return": 0, "benchmark_return": 0, "excess_return": 0,
                    "calmar": 0, "max_drawdown": 0, "annual_vol": 0, "periods_per_year": ppy}

        # 组合总收益
        total_return = (1 + port_year).prod() - 1

        # 基准收益
        idx_year = self._index_df[
            (self._index_df["trade_date"] >= f"{year}-01-01") &
            (self._index_df["trade_date"] <= f"{year}-12-31")
        ].copy()

        if len(idx_year) > 0:
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

        # Calmar (年化收益 / 最大回撤)
        calmar = total_return * ppy / len(port_year) / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "annual_vol": annual_vol,
            "periods_per_year": ppy,
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

    def validate_factor(
        self,
        factor_name: str,
        formula: str,
        skip_simplify: bool = False,
    ) -> S5Result:
        """
        对单个因子执行完整 S5 验证。
        
        v0.6.1: skip_simplify=True 时跳过规则6 (过度简化拦截), 用于 seed 重检。

        Returns:
          S5Result with passed=True if:
            - excess_2025 > 0 AND excess_2026 > 0
            - calmar_2025 > 1.0 AND calmar_2026 > 1.0
        """
        result = S5Result(factor_name=factor_name, formula=formula[:200])
        t0 = datetime.now()

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
        sanity_reason = self._check_formula_sanity(formula, factor_name,
                                                   skip_simplify=skip_simplify)
        if sanity_reason:
            result.reason = f"🧹 公式合理性拦截: {sanity_reason}"
            return result

        # 评估因子公式
        factor_df = self.evaluate_factor_formula(formula, factor_name)
        if factor_df is None or len(factor_df) == 0:
            result.reason = "因子公式评估无有效数据"
            return result

        # v0.7 P-025: 常数/低基数因子拦截 —
        # 常数因子选股 = 随机组合, 回测结果纯噪声 (Forge Round1 实证: sign(volume) 恒为 1)。
        try:
            fvals = pd.to_numeric(factor_df["factor_value"], errors="coerce").dropna()
            if len(fvals) == 0:
                result.reason = "因子值全为 NaN"
                return result
            n_unique = fvals.round(12).nunique()
            if n_unique <= 5:
                result.reason = (f"🧹 常数因子拦截: 有效值仅 {n_unique} 个离散水平, "
                                 f"选股无区分度")
                return result
        except Exception:
            pass

        # v0.7 P-025: 价格水平代理拦截 —
        # ts_sum(close,20) 类因子 ≈ 按价格水平选股 (高价股组合), 非 alpha。
        # Forge Round1b 实证: 该因子 S5 通过但属价格代理假阳性。
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
                    result.reason = (f"🧹 价格水平代理拦截: 因子与收盘价 rank 相关 "
                                     f"中位数 {med:.3f} > 0.95 (非 alpha)")
                    return result
        except Exception:
            pass

        # v0.9.2 P-20260823-005: 逐日 IC 序列写入 DecayMonitor 缓存
        # (修复: cache_ic_series 此前零调用, DecayMonitor 长期空转于合成残留数据)
        try:
            ic_series = self._daily_ic_series(factor_df)
            if len(ic_series) >= 20:
                from decay_monitor import get_decay_monitor
                get_decay_monitor().update_ic_cache(factor_name, ic_series)
        except Exception:
            pass

        # 回测
        port_ret, _ = self.backtest_factor(factor_df)
        # v0.6.1: 收益序列附回 (S6 DSR / 增量门禁用)
        result.portfolio_returns = port_ret
        result.portfolio_returns_n = len(port_ret)
        
        # 分年度计算
        metrics_2025 = self.compute_metrics(port_ret, 2025)
        metrics_2026 = self.compute_metrics(port_ret, 2026)

        result.excess_2025 = metrics_2025["excess_return"]
        result.excess_2026 = metrics_2026["excess_return"]
        result.calmar_2025 = metrics_2025["calmar"]
        result.calmar_2026 = metrics_2026["calmar"]
        result.total_return_2025 = metrics_2025["total_return"]
        result.total_return_2026 = metrics_2026["total_return"]
        result.benchmark_return_2025 = metrics_2025["benchmark_return"]
        result.benchmark_return_2026 = metrics_2026["benchmark_return"]
        result.max_drawdown_2025 = metrics_2025["max_drawdown"]
        result.max_drawdown_2026 = metrics_2026["max_drawdown"]
        result.annual_vol_2025 = metrics_2025["annual_vol"]
        result.annual_vol_2026 = metrics_2026["annual_vol"]

        # S5 判定
        excess_ok = result.excess_2025 > self.excess_threshold and result.excess_2026 > self.excess_threshold
        calmar_ok = result.calmar_2025 > self.calmar_threshold and result.calmar_2026 > self.calmar_threshold

        if excess_ok and calmar_ok:
            result.passed = True
            result.reason = (
                f"✅ 两年正向过滤通过: "
                f"excess_25={result.excess_2025:.2%}, excess_26={result.excess_2026:.2%}, "
                f"calmar_25={result.calmar_2025:.2f}, calmar_26={result.calmar_2026:.2f}"
            )
        elif excess_ok and not calmar_ok:
            result.reason = (
                f"❌ Calmar不达标: "
                f"calmar_25={result.calmar_2025:.2f}, calmar_26={result.calmar_2026:.2f} (需 >{self.calmar_threshold})"
            )
        elif not excess_ok and calmar_ok:
            result.reason = (
                f"❌ 超额不达标: "
                f"excess_25={result.excess_2025:.2%}, excess_26={result.excess_2026:.2%} (需 >{self.excess_threshold:.0%})"
            )
        else:
            result.reason = (
                f"❌ 两项均不达标: "
                f"excess_25={result.excess_2025:.2%}, excess_26={result.excess_2026:.2%}, "
                f"calmar_25={result.calmar_2025:.2f}, calmar_26={result.calmar_2026:.2f}"
            )

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
            result = self.validate_factor(name, formula, skip_simplify=skip_simplify)
            results.append(result)

            if verbose and result.passed:
                print(f"    ✅ PASS: excess_25={result.excess_2025:.2%} excess_26={result.excess_2026:.2%} "
                      f"calmar_25={result.calmar_2025:.2f} calmar_26={result.calmar_2026:.2f}")
            elif verbose and not result.passed:
                print(f"    {result.reason[:100]}")

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
