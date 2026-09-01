# -*- coding: utf-8 -*-
"""
轻量级因子 IC 计算器 — 不依赖 S5JointFilter，直接读取 daily_prices.csv
==========================================================================

用于 S1 快速 IC 筛选，支持 pandas 风格和 Forge DSL 风格因子公式。

设计原则:
- 独立于 S5JointFilter（不走完整的 backtest 管线）
- 只做 Rank IC + ICIR 计算（Spearman cross-section）
- 小样本（200 stocks）快速计算，用于 S1 筛选门槛

用法:
    from factor_ic_computer import FactorICComputer
    comp = FactorICComputer()
    result = comp.compute(formula, factor_name)

    # result: {"ic": float, "icir": float, "n_days": int}
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Optional, List


RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PRICE_PATH = RAW_DIR / "daily_prices.csv"
MONEYFLOW_PATH = RAW_DIR / "moneyflow_daily.csv"
MARGIN_PATH = RAW_DIR / "margin_detail.csv"
TOP_LIST_PATH = RAW_DIR / "top_list.csv"
TOP_INST_PATH = RAW_DIR / "top_inst.csv"
HK_HOLD_PATH = RAW_DIR / "hk_hold.csv"
BASIC_PATH = RAW_DIR / "daily_basic.csv"
FINA_IND_PATH = RAW_DIR / "fina_indicator.csv"

# v0.5.4: 主动资金流字段 (merge 自 moneyflow_daily.csv, 供公式 eval 使用)
MONEYFLOW_FIELDS = [
    "buy_lg_vol", "sell_lg_vol", "buy_sm_vol", "sell_sm_vol",
    "buy_md_vol", "sell_md_vol", "buy_elg_vol", "sell_elg_vol",
    "net_mf_vol", "net_mf_amount",
]

# v0.9 P-20260814-001: 两融字段 (merge 自 margin_detail.csv, 存量数据至 2025-12-31)
MARGIN_FIELDS = ["rzye", "rqye", "rzmre", "rqyl", "rqchl"]

# v0.9 P-20260812-026: 龙虎榜字段 (top_list.csv 2023-01 起, 未上榜=0)
TOP_LIST_FIELDS = ["lhb_flag", "lhb_net_amount", "lhb_net_rate", "lhb_amount"]

# v0.9 P-20260812-026: 机构专用席位字段 (top_inst.csv, exalter 含"机构专用")
TOP_INST_FIELDS = ["lhb_inst_net_buy", "lhb_inst_buy", "lhb_inst_sell"]

# v0.9.2 P-20260823-001: 北向专用席位字段 (top_inst.csv, exalter 含"深股通专用|沪股通专用")
NORTH_INST_FIELDS = ["lhb_north_net_buy", "lhb_north_buy", "lhb_north_sell"]

# v0.9 P-20260814-002: 北向持股字段 (hk_hold.csv SH+SZ, 2024-12-31 停更 → NaN 保留)
NORTH_FIELDS = ["north_vol", "north_ratio"]

# v0.9.1 P-20260819-002: 分析师分歧字段 (report_rc.csv 中间表, 与 daily_factor_hypothesis 同口径)
ANALYST_FIELDS = ["eps_disp", "tp_disp", "rating_disp", "n_cover"]

# v0.9.1 P-20260819-003: 行业 peer 收益 (申万 L1 成分等权, 个股化展开)
INDUSTRY_PEER_FIELDS = ["industry_ret_peer"]

# v0.10.1 P-20260827-004: 行业 L1 代码静态映射 (分类变量, 供行业条件化公式; NA=无归属)
INDUSTRY_CODE_FIELDS = ["industry_l1"]

# v0.10 P-20260825-001: 估值字段 (daily_basic 慢变量, 按股 ffill; 日历与 daily_prices 错位1-2天)
# v0.10.1 P-20260827-004: + circ_mv 流通市值 (规模因子行业/市值分段条件化用)
VALUATION_FIELDS = ["pe_ttm", "pb", "dv_ratio", "circ_mv"]

# v0.10 P-20260825-001: 财务质量字段 (fina_indicator, ann_date 公告日 asof merge 无前视)
FINA_FIELDS = ["roe_dt", "tr_yoy", "netprofit_yoy"]


class FactorICComputer:
    """轻量级因子 IC 计算器（不依赖 S5 基础设施）"""

    def __init__(
        self,
        price_path: Path = None,
        sample_stocks: int = None,
        forward_period: int = 5,
        min_stocks_per_day: int = 30,
        lookback_days: int = 270,
    ):
        self.price_path = price_path or PRICE_PATH
        # v0.9.2: None = 全市场向量化 (消除抽样口径差); 显式传 int 仍可抽样降本
        self.sample_stocks = sample_stocks
        self.forward_period = forward_period
        self.min_stocks_per_day = min_stocks_per_day
        # v0.9.2: lookback_days 对齐本地 daily_factor_hypothesis.load_data(n_days=270)
        # (消除时间窗口口径差: 因子时效性导致长窗 IC 与近窗符号可相反)
        self.lookback_days = lookback_days
        self._price_df: Optional[pd.DataFrame] = None
        self._universe: List[str] = []
        self._all_dates: pd.DatetimeIndex = None
        self._wide_cache: Dict[str, pd.DataFrame] = {}  # v0.9.2: 字段→宽表缓存
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self.price_path.exists()

    def _load_data(self):
        """加载 daily_prices.csv 并预处理"""
        if self._loaded:
            return

        if not self.price_path.exists():
            raise FileNotFoundError(f"Price data not found: {self.price_path}")

        print(f"  [IC Computer] Loading {self.price_path}...")
        df = pd.read_csv(self.price_path, low_memory=False)

        # 统一列名
        col_map = {
            "vol": "volume", "turnover_vol": "volume",
            "amt": "amount", "turnover_val": "amount",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # v0.9.2 修复: trade_date 混有 YYYYMMDD 与 YYYY-MM-DD 两种格式 (增量采集漂移)
        # 无 format 时 10 位格式解析成 NaT → 近期日期丢失 + 重复行 (与本地 load_data 对齐)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format='mixed', errors="coerce")
        df = df.dropna(subset=["trade_date"])
        # 2026-08-31 数据修复防线: 原文件曾含 6.4M 重复行 (两批拼接),
        # 加载时强制去重 — 修复后文件无重复, 此行为零成本保险 (keep=last)。
        _n0 = len(df)
        df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        if len(df) < _n0:
            print(f"  [IC Computer] ⚠️ 去重: {_n0 - len(df):,} 重复行已剔除 "
                  f"(原始文件仍有重复, 建议重采)")
        # v0.9.2: lookback_days 对齐本地检验窗口 (默认 270 自然日 ≈ 190 交易日)
        if self.lookback_days is not None:
            start_dt = df["trade_date"].max() - pd.Timedelta(days=self.lookback_days)
            df = df[df["trade_date"] >= start_dt]
        else:
            df = df[df["trade_date"] >= "2020-01-01"]

        # 补全可选列
        for col in ["volume", "amount"]:
            if col not in df.columns:
                df[col] = 0

        # _p 后缀别名（pandas eval 兼容）
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[f"{col}_p"] = df[col]

        # 衍生字段
        if "open" in df.columns and "close" in df.columns:
            df["overnight"] = df["open"] / df.groupby("ts_code")["close"].shift(1) - 1
            df["overnight_p"] = df["overnight"]

        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            df["amplitude"] = (df["high"] - df["low"]) / df.groupby("ts_code")["close"].shift(1)
            df["amplitude_p"] = df["amplitude"]

        if "close" in df.columns:
            df["returns"] = df.groupby("ts_code")["close"].pct_change()
            df["returns_p"] = df["returns"]

        if "volume" in df.columns and "close" in df.columns:
            df["turnover"] = df["volume"] / df["close"].clip(lower=0.01)
            df["turnover_p"] = df["turnover"]

        if "high" in df.columns and "low" in df.columns:
            df["hl_ratio"] = df["high"] / df["low"].clip(lower=0.01)
            df["hl_ratio_p"] = df["hl_ratio"]

        # v0.5.4: merge 主动资金流数据 (buy_lg_vol/sell_lg_vol/... 修复 NameError)
        if MONEYFLOW_PATH.exists():
            try:
                mf = pd.read_csv(MONEYFLOW_PATH, low_memory=False)
                # trade_date 为 YYYYMMDD 数字格式 (如 20260116), 直接 to_datetime 会解析成
                # 1970 年 epoch 导致与价格表 merge 全部错位 → 需显式 format
                try:
                    mf["trade_date"] = pd.to_datetime(
                        mf["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                except Exception:
                    mf["trade_date"] = pd.to_datetime(mf["trade_date"], errors='coerce')
                mf_cols = ["ts_code", "trade_date"] + [
                    c for c in MONEYFLOW_FIELDS if c in mf.columns
                ]
                if len(mf_cols) > 2:
                    mf = mf[mf_cols].dropna(subset=["trade_date"])
                    df = df.merge(mf, on=["ts_code", "trade_date"], how="left")
                    for col in MONEYFLOW_FIELDS:
                        if col in df.columns:
                            df[col] = df[col].fillna(0)
                            df[f"{col}_p"] = df[col]
                    print(f"  [IC Computer] Loaded moneyflow: "
                          f"{len([c for c in MONEYFLOW_FIELDS if c in df.columns])} fields")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ moneyflow load failed: {e}")

        # v0.9 P-20260814-001: merge 两融数据 (rzye/rqye/rzmre/rqyl/rqchl)
        if MARGIN_PATH.exists():
            try:
                mg = pd.read_csv(MARGIN_PATH, low_memory=False)
                mg["trade_date"] = pd.to_datetime(
                    mg["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                mg_cols = ["ts_code", "trade_date"] + [
                    c for c in MARGIN_FIELDS if c in mg.columns
                ]
                if len(mg_cols) > 2:
                    mg = mg[mg_cols].dropna(subset=["trade_date"])
                    df = df.merge(mg, on=["ts_code", "trade_date"], how="left")
                    for col in MARGIN_FIELDS:
                        if col in df.columns:
                            df[f"{col}_p"] = df[col]
                    print(f"  [IC Computer] Loaded margin: "
                          f"{len([c for c in MARGIN_FIELDS if c in df.columns])} fields "
                          f"(至 {mg['trade_date'].max().date()})")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ margin load failed: {e}")

        # v0.9 P-20260812-026: merge 龙虎榜 top_list (事件稀疏, 未上榜填 0)
        if TOP_LIST_PATH.exists():
            try:
                tl = pd.read_csv(TOP_LIST_PATH, low_memory=False)
                tl["trade_date"] = pd.to_datetime(
                    tl["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                for c in ["net_amount", "net_rate", "amount"]:
                    tl[c] = pd.to_numeric(tl[c], errors="coerce")
                tl_agg = tl.groupby(["ts_code", "trade_date"], as_index=False).agg(
                    lhb_flag=("ts_code", "size"),
                    lhb_net_amount=("net_amount", "sum"),
                    lhb_net_rate=("net_rate", "sum"),
                    lhb_amount=("amount", "sum"),
                )
                df = df.merge(tl_agg, on=["ts_code", "trade_date"], how="left")
                for col in TOP_LIST_FIELDS:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                        df[f"{col}_p"] = df[col]
                print(f"  [IC Computer] Loaded top_list: {len(tl_agg)} stock-days "
                      f"(至 {tl['trade_date'].max().date()})")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ top_list load failed: {e}")

        # v0.9 P-20260812-026: merge 机构专用席位 top_inst (未上榜填 0)
        if TOP_INST_PATH.exists():
            try:
                ti = pd.read_csv(TOP_INST_PATH, low_memory=False)
                ti["trade_date"] = pd.to_datetime(
                    ti["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                inst = ti[ti["exalter"].str.contains("机构专用", na=False)]
                for c in ["buy", "sell", "net_buy"]:
                    inst[c] = pd.to_numeric(inst[c], errors="coerce")
                inst_agg = inst.groupby(["ts_code", "trade_date"], as_index=False).agg(
                    lhb_inst_net_buy=("net_buy", "sum"),
                    lhb_inst_buy=("buy", "sum"),
                    lhb_inst_sell=("sell", "sum"),
                )
                df = df.merge(inst_agg, on=["ts_code", "trade_date"], how="left")
                for col in TOP_INST_FIELDS:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                        df[f"{col}_p"] = df[col]
                print(f"  [IC Computer] Loaded top_inst(机构专用): {len(inst_agg)} stock-days")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ top_inst load failed: {e}")

        # v0.9.2 P-20260823-001: merge 北向专用席位 top_inst (深股通/沪股通专用, 未上榜填 0)
        if TOP_INST_PATH.exists():
            try:
                ti = pd.read_csv(TOP_INST_PATH, low_memory=False)
                ti["trade_date"] = pd.to_datetime(
                    ti["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                north = ti[ti["exalter"].str.contains(
                    "深股通专用|沪股通专用", na=False, regex=True)]
                for c in ["buy", "sell", "net_buy"]:
                    north[c] = pd.to_numeric(north[c], errors="coerce")
                north_agg = north.groupby(["ts_code", "trade_date"], as_index=False).agg(
                    lhb_north_net_buy=("net_buy", "sum"),
                    lhb_north_buy=("buy", "sum"),
                    lhb_north_sell=("sell", "sum"),
                )
                df = df.merge(north_agg, on=["ts_code", "trade_date"], how="left")
                for col in NORTH_INST_FIELDS:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                        df[f"{col}_p"] = df[col]
                print(f"  [IC Computer] Loaded top_inst(北向专用): {len(north_agg)} stock-days")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ top_inst(北向专用) load failed: {e}")

        # v0.9 P-20260814-002: merge 北向持股 hk_hold (2024-12-31 停更 → NaN 保留,
        # 因子公式里 rolling 会自动跳过 NaN 窗口; 有效 IC 窗口 2020-2024)
        if HK_HOLD_PATH.exists():
            try:
                hh = pd.read_csv(HK_HOLD_PATH, low_memory=False)
                hh = hh[hh["exchange"].isin(["SH", "SZ"])]
                hh["trade_date"] = pd.to_datetime(
                    hh["trade_date"].astype(str), format='%Y%m%d', errors='coerce')
                for c in ["vol", "ratio"]:
                    hh[c] = pd.to_numeric(hh[c], errors="coerce")
                hh_agg = hh.groupby(["ts_code", "trade_date"], as_index=False).agg(
                    north_vol=("vol", "sum"),
                    north_ratio=("ratio", "sum"),
                )
                df = df.merge(hh_agg, on=["ts_code", "trade_date"], how="left")
                for col in NORTH_FIELDS:
                    if col in df.columns:
                        df[f"{col}_p"] = df[col]  # 不 fillna: 缺失=停更/未持有, 非零
                print(f"  [IC Computer] Loaded hk_hold(北向): {len(hh_agg)} stock-days "
                      f"(至 {hh['trade_date'].max().date()})")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ hk_hold load failed: {e}")

        # v0.9.1 P-20260819-002/003: 接入 data_loader_ext 扩展数据层
        # (分析师分歧 report_rc 中间表 + 申万 L1 行业 peer 收益)
        # 与 daily_factor_hypothesis 检验环境同口径, 消除 ralph 侧 NameError。
        try:
            import sys as _sys
            _scripts_dir = Path(__file__).parent.parent.parent / "scripts"
            if str(_scripts_dir) not in _sys.path:
                _sys.path.insert(0, str(_scripts_dir))
            from data_loader_ext import (
                load_analyst_forecast_wide, load_industry_wide)

            daily_index = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
            daily_columns = list(df["ts_code"].unique())

            def _wide_to_long(w, colname):
                """宽表 (dates × stocks) → 长表 [ts_code, trade_date, colname]"""
                if w is None or getattr(w, "size", 0) == 0:
                    return None
                long = (w.rename_axis("trade_date")
                         .rename_axis("ts_code", axis=1)
                         .stack()
                         .rename(colname)
                         .reset_index())
                return long if len(long) > 0 else None

            # 轻量列注入: 用 keys-only 临时表 merge, 避免直接 merge 13M×73列
            # float64 主表 (consolidate 需 7.26GiB 连续内存, Windows 必爆)。
            keys_df = df[["ts_code", "trade_date"]].copy()

            def _inject_column(long, colname):
                m = keys_df.merge(long, on=["ts_code", "trade_date"], how="left")
                df.drop(columns=[colname], errors="ignore", inplace=True)
                df[colname] = m[colname].values

            # 分析师分歧 (eps_disp/tp_disp/rating_disp/n_cover)
            an_data = load_analyst_forecast_wide(daily_index, daily_columns)
            n_an = 0
            for name, w in zip(ANALYST_FIELDS, an_data):
                long = _wide_to_long(w, name)
                if long is None:
                    continue
                _inject_column(long, name)
                n_an += 1
            if n_an:
                cov = int(df["eps_disp"].notna().sum()) if "eps_disp" in df.columns else 0
                print(f"  [IC Computer] Loaded analyst(分析师分歧): {n_an} fields "
                      f"(eps_disp 有效 {cov} stock-days)")

            # 行业 peer 收益 (industry_ret_peer, 已按申万 L1 个股化展开)
            peer_w = load_industry_wide(daily_index, daily_columns)
            long = _wide_to_long(peer_w, "industry_ret_peer")
            if long is not None:
                _inject_column(long, "industry_ret_peer")
                print(f"  [IC Computer] Loaded industry_peer(行业L1等权): "
                      f"{len(long)} stock-days")

            # P-20260827-004: 行业 L1 代码静态映射 (industry_l1, str 分类, 无时变)
            _mem_path = RAW_DIR / "sw_index_member_all.csv"
            if _mem_path.exists():
                try:
                    _mem = pd.read_csv(_mem_path, dtype={'ts_code': str, 'l1_code': str},
                                       usecols=['ts_code', 'l1_code'])
                    _code2l1 = dict(zip(_mem['ts_code'], _mem['l1_code']))
                    df["industry_l1"] = df["ts_code"].map(_code2l1).fillna("NA")
                    _n_mapped = int((df["industry_l1"] != "NA").sum())
                    print(f"  [IC Computer] Loaded industry_l1(申万L1代码): "
                          f"{len(_mem)} 只映射, 覆盖 {_n_mapped} stock-days")
                    del _mem
                except Exception as _ie:
                    print(f"  [IC Computer] ⚠️ industry_l1 加载失败: {_ie}")
            del keys_df
        except ImportError as e:
            print(f"  [IC Computer] ⚠️ data_loader_ext 不可用: {e} "
                  f"(分析师分歧/行业 peer 未接入)")
        except Exception as e:
            print(f"  [IC Computer] ⚠️ 扩展数据层加载失败: {e}")

        # v0.10 P-20260825-001: 估值字段 (daily_basic, keys-only 注入, 慢变量按股 ffill)
        if BASIC_PATH.exists():
            try:
                _val_cols = [c for c in VALUATION_FIELDS]
                ba = pd.read_csv(
                    BASIC_PATH,
                    usecols=lambda c: c in ["ts_code", "trade_date"] + _val_cols,
                    low_memory=False)
                ba["trade_date"] = pd.to_datetime(
                    ba["trade_date"].astype(str), format='mixed', errors='coerce')
                ba = ba.dropna(subset=["trade_date", "ts_code"])
                ba = ba.drop_duplicates(["ts_code", "trade_date"], keep="last")
                _ba_cols = ["ts_code", "trade_date"] + [c for c in _val_cols if c in ba.columns]
                _keys2 = df[["ts_code", "trade_date"]].copy()
                df = df.drop(columns=[c for c in _val_cols if c in df.columns],
                             errors="ignore")
                _m2 = _keys2.merge(ba[_ba_cols], on=["ts_code", "trade_date"],
                                   how="left")
                for col in _val_cols:
                    if col in ba.columns:
                        df[col] = _m2[col].values
                        df[f"{col}_p"] = df[col]
                del _keys2, _m2, ba
                print(f"  [IC Computer] Loaded valuation(daily_basic): "
                      f"{len([c for c in _val_cols if c in df.columns])} fields")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ daily_basic load failed: {e}")

        # v0.10 P-20260825-001: 财务质量字段 (fina_indicator, ann_date 公告日
        # asof → 宽表 reindex 价格日历 → ffill, 公告日之前不可见 = 无前视)
        if FINA_IND_PATH.exists():
            try:
                _fi_cols = [c for c in FINA_FIELDS]
                fi = pd.read_csv(
                    FINA_IND_PATH,
                    usecols=lambda c: c in ["ts_code", "ann_date"] + _fi_cols,
                    low_memory=False)
                fi["ann_date"] = pd.to_datetime(
                    fi["ann_date"].astype(str).str.replace(r"\.0$", "", regex=True),
                    format='mixed', errors='coerce')
                fi = fi.dropna(subset=["ann_date", "ts_code"])
                fi = fi.drop_duplicates(["ts_code", "ann_date"], keep="last")
                fi = fi.sort_values(["ts_code", "ann_date"])
                _daily_idx = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
                for col in _fi_cols:
                    if col not in fi.columns:
                        continue
                    w = fi.pivot_table(index="ann_date", columns="ts_code",
                                       values=col, aggfunc="last")
                    w = w.reindex(_daily_idx).ffill()
                    long = (w.rename_axis("trade_date").rename_axis("ts_code", axis=1)
                             .stack().rename(col).reset_index())
                    _keys3 = df[["ts_code", "trade_date"]].copy()
                    df = df.drop(columns=[col], errors="ignore")
                    _m3 = _keys3.merge(long, on=["ts_code", "trade_date"], how="left")
                    df[col] = _m3[col].values
                    df[f"{col}_p"] = df[col]
                    del _keys3, _m3, long, w
                del fi
                print(f"  [IC Computer] Loaded fina quality(ann_date asof): "
                      f"{len([c for c in _fi_cols if c in df.columns])} fields")
            except Exception as e:
                print(f"  [IC Computer] ⚠️ fina_indicator load failed: {e}")

        # v0.10: 慢变量按股 ffill (daily_basic 月末缺日 / 财报公告间隔)
        _slow_cols = [c for c in VALUATION_FIELDS + FINA_FIELDS if c in df.columns]
        if _slow_cols:
            df = df.sort_values(["ts_code", "trade_date"])
            df[_slow_cols] = df.groupby("ts_code")[_slow_cols].ffill()

        # v0.5.5 内存保护 (2026-08-15): 13.3M行 × ~73列 float64 在 pandas consolidate
        # 时需一次性 7.25 GiB 连续内存, Windows 下大块连续分配易失败 (_ArrayMemoryError)
        # → float64 转 float32 减半 (IC 为 rank 相关, 精度无影响)
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)

        # v0.9.2: merge 后统一去重 (历史采集 + 多源 merge 可能产生重复 ts_code+trade_date)
        df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

        self._price_df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # v0.9.2: universe 全市场向量化 (消除抽样口径差)
        if "ts_code" in df.columns:
            all_codes = list(df["ts_code"].unique())
            if self.sample_stocks is None or self.sample_stocks >= len(all_codes):
                self._universe = all_codes  # 全市场
            else:
                self._universe = list(np.random.RandomState(42).choice(
                    all_codes, size=min(self.sample_stocks, len(all_codes)), replace=False
                ))
        self._all_dates = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
        self._wide_cache.clear()

        self._loaded = True
        mode = "全市场" if self.sample_stocks is None else f"抽样{self.sample_stocks}"
        print(f"    Loaded {len(df)} rows, {df['ts_code'].nunique()} stocks, "
              f"universe={len(self._universe)} stocks ({mode}, 向量化)")

    def compute(
        self, formula: str, factor_name: str = "unknown",
        return_series: bool = False,
        also_lag1: bool = False,
    ) -> Dict[str, float]:
        """
        计算因子的 Rank IC (Spearman)。

        Returns:
            {"ic": float, "icir": float, "n_days": int}
            return_series=True 时额外返回 {"ic_series": [float, ...]}
            (P-20260828-004: 供 S5 block bootstrap 置信区间影子模式使用)
            also_lag1=True 时额外返回 {"ic_lag1": float, "delta_lag": float,
                                       "n_days_lag1": int}
            (P-20260828-001: Δlag = RankIC - RankIC_lag1, 滞后衰减影子模式;
             lag1 = 因子值按 ts_code shift(+1) 个交易日, 即用昨日信号配今日起的未来收益)
        """
        if not self._loaded:
            self._load_data()

        if self._price_df is None or len(self._universe) == 0:
            out = {"ic": 0.0, "icir": 0.0, "n_days": 0}
            if return_series:
                out["ic_series"] = []
            if also_lag1:
                out["ic_lag1"] = 0.0
                out["delta_lag"] = 0.0
                out["n_days_lag1"] = 0
            return out

        # 评估因子公式在 universe 上的值
        factor_df = self._eval_formula(formula, factor_name)
        if factor_df is None or len(factor_df) == 0:
            out = {"ic": 0.0, "icir": 0.0, "n_days": 0}
            if return_series:
                out["ic_series"] = []
            if also_lag1:
                out["ic_lag1"] = 0.0
                out["delta_lag"] = 0.0
                out["n_days_lag1"] = 0
            return out

        # 获取价格数据用于计算前向收益
        price_data = self._price_df[self._price_df["ts_code"].isin(self._universe)][
            ["ts_code", "trade_date", "close"]
        ].drop_duplicates(subset=["ts_code", "trade_date"])

        # Merge factor values with price
        merged = factor_df.merge(price_data, on=["ts_code", "trade_date"], how="inner")
        merged = merged.sort_values(["ts_code", "trade_date"])
        merged["fwd_return"] = (
            merged.groupby("ts_code")["close"].shift(-self.forward_period) / merged["close"] - 1
        )
        merged = merged.dropna(subset=["factor_value", "fwd_return"])

        # 逐日 Rank IC
        ic_series = []
        for date, group in merged.groupby("trade_date"):
            if len(group) < self.min_stocks_per_day:
                continue
            rank_factor = group["factor_value"].rank()
            rank_fwd = group["fwd_return"].rank()
            ic = rank_factor.corr(rank_fwd)
            if not np.isnan(ic):
                ic_series.append(ic)

        if not ic_series:
            out = {"ic": 0.0, "icir": 0.0, "n_days": 0}
            if return_series:
                out["ic_series"] = []
            if also_lag1:
                out["ic_lag1"] = 0.0
                out["delta_lag"] = 0.0
                out["n_days_lag1"] = 0
            return out

        ic_arr = np.array(ic_series)
        mean_ic = float(np.nanmean(ic_arr))
        std_ic = float(np.nanstd(ic_arr))
        # v0.9.2: ICIR 取绝对值, 对齐本地 compute_ic_icir 口径 (ICIR 定义非负)
        icir = abs(mean_ic / std_ic) if std_ic > 0 else 0.0

        out = {
            "ic": round(mean_ic, 6),
            "icir": round(icir, 4),
            "n_days": len(ic_series),
        }
        if return_series:
            out["ic_series"] = [float(v) for v in ic_series]

        # ── P-20260828-001: Δlag 滞后衰减 (影子模式) ──
        # 信号右移一天: 用昨日因子值预测今日起的未来收益。
        # 在同日期集合上对比 当期IC vs lag1IC, Δlag = 当期IC - lag1IC。
        # 正值 = 信号延迟一天后预测力下降 (衰减); 负值 = 延迟反而更好 (异常)。
        if also_lag1:
            lag_merged = merged.copy()
            lag_merged["factor_value_lag1"] = (
                lag_merged.groupby("ts_code")["factor_value"].shift(1)
            )
            lag_merged = lag_merged.dropna(subset=["factor_value_lag1", "fwd_return"])
            ic_lag1_series = []
            ic_same_series = []  # 同日期集合上的当期 IC (公平对比)
            for date, group in lag_merged.groupby("trade_date"):
                if len(group) < self.min_stocks_per_day:
                    continue
                ic_lag = group["factor_value_lag1"].rank().corr(
                    group["fwd_return"].rank())
                ic_same = group["factor_value"].rank().corr(
                    group["fwd_return"].rank())
                if not np.isnan(ic_lag) and not np.isnan(ic_same):
                    ic_lag1_series.append(ic_lag)
                    ic_same_series.append(ic_same)
            if ic_lag1_series:
                mean_lag1 = float(np.nanmean(ic_lag1_series))
                mean_same = float(np.nanmean(ic_same_series))
                out["ic_lag1"] = round(mean_lag1, 6)
                out["delta_lag"] = round(mean_same - mean_lag1, 6)
                out["n_days_lag1"] = len(ic_lag1_series)
            else:
                out["ic_lag1"] = 0.0
                out["delta_lag"] = 0.0
                out["n_days_lag1"] = 0
        return out

    def _eval_formula(self, formula: str, factor_name: str) -> Optional[pd.DataFrame]:
        """
        评估因子公式在 universe 上的原始值。

        返回 DataFrame: columns=[ts_code, trade_date, factor_value]
        """
        import warnings
        warnings.filterwarnings("ignore")

        # v0.5.4: 自动加载数据 (修复直接调用 _eval_formula 时 _price_df=None 崩溃)
        if not self._loaded:
            self._load_data()

        clean = formula.strip()

        # ── v0.5.4: 多行公式支持 (因子池 133/280 为多行迷你程序) ──
        #   中间行用 exec 逐行执行 (赋值进入命名空间), 最后一行作为结果表达式
        # P-20260831-002: 剥离 import 行 — safe_builtins 禁 __import__,
        #   "import numpy as np" 行导致 ImportError: __import__ not found
        #   (price_volume_coupling_decay / earnings_cross_section_decay 挂起根因);
        #   np/pd 已在 base_context 提供, 剥离后语义不变
        lines = [ln.strip() for ln in clean.split("\n")
                 if ln.strip() and not ln.strip().startswith("#")
                 and not re.match(r'^(import\s|from\s+\S+\s+import)', ln.strip())]
        body_lines = []
        if len(lines) > 1:
            body_lines = lines[:-1]
            final_line = lines[-1]
            m = re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$', final_line)
            if m:
                final_line = m.group(1).strip()
            clean = final_line
        else:
            # 去掉赋值
            if "=" in clean and not any(op in clean for op in ["<=", ">=", "==", "!="]):
                parts = clean.split("=", 1)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', parts[0].strip()):
                    clean = parts[1].strip()

        # DSL → pandas 自动转换 (仅对最终表达式)
        clean = self._normalize_to_pandas(clean)
        if not clean:
            return None

        # v0.9.4 rank 语义修正 (2026-08-21 rmb 因子审计):
        # 宽表 DataFrame 上裸 .rank(pct=True) 默认 axis=0 = 沿时间轴排名,
        # 用整个回看窗口(含未来)计算每天分位 → 前视偏差。
        # 实证: rmb_appreciation_sensitive_spread 裸 rank ICIR=1.526 假阳性,
        # axis=1 修正后 ICIR=0.174。例外: .rolling(N).rank(...) 保持原样。
        # v0.9.5: 负向后顾变长 \d+ 会导致 re.compile 抛异常 (定宽限制),
        # 改为「先保护 rolling(N).rank 占位 → 裸 rank 改 axis=1 → 还原」三段式。
        import re as _re
        # v0.6.2 (2026-08-29): rolling 括号内允许任意 kwargs (min_periods 等),
        # 原 \d+ 匹配不了 rolling(252, min_periods=60) → 裸 rank 重写误伤
        # Rolling.rank() → TypeError: Rolling.rank() got an unexpected keyword
        # argument 'axis' (Forge 截面 rank 旧翻译的产物, Round5 实证)。
        # 捕获组 (group 1) 保留完整 rolling(...) 调用, 供占位还原。
        _roll_rank_re = _re.compile(r'(\.rolling\([^)]*\))\.rank\(pct\s*=\s*True\)')
        _plain_rank_re = _re.compile(r'\.rank\(pct\s*=\s*True\)')
        _holders = []
        clean = _roll_rank_re.sub(
            lambda m: _holders.append(m.group(0)) or f'{m.group(1)}.__RRK{len(_holders)-1}__',
            clean)
        clean = _plain_rank_re.sub('.rank(pct=True, axis=1)', clean)
        for _i, _h in enumerate(_holders):
            clean = clean.replace(f'__RRK{_i}__', 'rank(pct=True)')

        # v0.9.2: 向量化宽表 eval (逐股 eval 已废弃, 全市场一次运算)
        safe_builtins = {
            'range': range, 'len': len, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'str': str, 'bool': bool,
            'True': True, 'False': False, 'None': None,
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sum': sum, 'zip': zip, 'sorted': sorted,
            'print': print, 'isinstance': isinstance,
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        }
        base_context = {
            "np": np, "pd": pd,
            "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
            "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            "where": np.where, "clip": np.clip,
            "range": range, "len": len, "int": int, "float": float,
            "list": list, "dict": dict, "tuple": tuple,
        }

        field_cols = [
            "close_p", "open_p", "high_p", "low_p", "volume_p", "amount_p",
            "overnight", "overnight_p", "amplitude", "amplitude_p",
            "returns", "returns_p", "turnover", "turnover_p",
            "hl_ratio", "hl_ratio_p",
        ] + [f for f in MONEYFLOW_FIELDS] + [f"{f}_p" for f in MONEYFLOW_FIELDS] \
          + [f for f in MARGIN_FIELDS] + [f"{f}_p" for f in MARGIN_FIELDS] \
          + [f for f in TOP_LIST_FIELDS] + [f"{f}_p" for f in TOP_LIST_FIELDS] \
          + [f for f in TOP_INST_FIELDS] + [f"{f}_p" for f in TOP_INST_FIELDS] \
          + [f for f in NORTH_INST_FIELDS] + [f"{f}_p" for f in NORTH_INST_FIELDS] \
          + [f for f in NORTH_FIELDS] + [f"{f}_p" for f in NORTH_FIELDS] \
          + [f for f in ANALYST_FIELDS] + [f"{f}_p" for f in ANALYST_FIELDS] \
          + [f for f in INDUSTRY_PEER_FIELDS] + [f"{f}_p" for f in INDUSTRY_PEER_FIELDS] \
          + [f for f in INDUSTRY_CODE_FIELDS] \
          + [f for f in VALUATION_FIELDS] + [f"{f}_p" for f in VALUATION_FIELDS] \
          + [f for f in FINA_FIELDS] + [f"{f}_p" for f in FINA_FIELDS]
        suffix_map = {
            "close": "close_p", "open": "open_p", "high": "high_p",
            "low": "low_p", "volume": "volume_p", "amount": "amount_p",
        }
        # v0.5.4: moneyflow 字段直接以原名暴露 (buy_lg_vol 等)
        for f in MONEYFLOW_FIELDS:
            suffix_map[f] = f
        # v0.9 P-20260814-001: margin 字段直接以原名暴露 (rzye 等)
        for f in MARGIN_FIELDS:
            suffix_map[f] = f
        # v0.9 P-20260812-026: 龙虎榜字段直接以原名暴露 (lhb_* 等)
        for f in TOP_LIST_FIELDS + TOP_INST_FIELDS:
            suffix_map[f] = f
        # v0.9.2 P-20260823-001: 北向专用席位字段直接以原名暴露 (lhb_north_*)
        for f in NORTH_INST_FIELDS:
            suffix_map[f] = f
        # v0.9 P-20260814-002: 北向字段直接以原名暴露 (north_* 等)
        for f in NORTH_FIELDS:
            suffix_map[f] = f
        # v0.9.1 P-20260819-002/003: 分析师分歧 + 行业 peer 直接以原名暴露
        for f in ANALYST_FIELDS + INDUSTRY_PEER_FIELDS:
            suffix_map[f] = f
        # v0.10.1 P-20260827-004: 行业 L1 代码直接以原名暴露 (industry_l1)
        for f in INDUSTRY_CODE_FIELDS:
            suffix_map[f] = f
        # v0.10 P-20260825-001: 估值 + 财务质量字段直接以原名暴露
        for f in VALUATION_FIELDS + FINA_FIELDS:
            suffix_map[f] = f

        # v0.9.2 向量化: 构造宽表求值环境 {字段名: DataFrame(index=日期, columns=股票)}
        # 宽表上 .shift/.rolling/.pct_change 沿行(日期)方向, 天然等价于逐股时序。
        wide_env = {}
        for name in field_cols:
            base = name[:-2] if name.endswith("_p") else name
            if base in wide_env:
                wide_env[name] = wide_env[base]
                continue
            w = self._get_wide_field(base)
            if w is not None:
                wide_env[base] = w
                wide_env[name] = w
        for short, full in suffix_map.items():
            if full in wide_env and short not in wide_env:
                wide_env[short] = wide_env[full]

        full_globals = {"__builtins__": safe_builtins, **base_context, **wide_env}

        # P-20260901-002: df['field'] 宽表引用自动改写 → field
        #   (volume_price_microstructure_entropy_regime 等 4 因子挂起根因:
        #   向量化环境无 df 变量; 宽表列名即字段名, df['close']→close 语义等价)
        _df_col_re = re.compile(r"""df\s*\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]""")

        def _df_rewrite(s: str) -> str:
            return _df_col_re.sub(
                lambda m: m.group(1) if m.group(1) in wide_env else m.group(0), s)

        body_lines = [_df_rewrite(ln) for ln in body_lines]
        clean = _df_rewrite(clean)

        try:
            # 中间赋值行 (多行公式) 写入命名空间
            for ln in body_lines:
                exec(ln, full_globals)
            result = eval(clean, full_globals, {})

            if isinstance(result, pd.DataFrame):
                # index=日期, columns=股票 → 长表 [trade_date, ts_code, factor_value]
                factor_df = (result.rename_axis("trade_date")
                             .rename_axis("ts_code", axis=1)
                             .stack()
                             .rename("factor_value")
                             .reset_index())
                factor_df = factor_df[["ts_code", "trade_date", "factor_value"]]
            elif isinstance(result, (int, float, np.floating, np.integer)):
                # 常数因子: 广播到 universe (后续 IC 恒 0, 无意义但不出错)
                factor_df = pd.DataFrame({
                    "ts_code": np.repeat(self._universe, len(self._all_dates)),
                    "trade_date": np.tile(self._all_dates, len(self._universe)),
                    "factor_value": float(result),
                })
            elif isinstance(result, pd.Series):
                # index=日期 (跨股票聚合) → 广播到全市场
                wide = result.to_frame().reindex(columns=self._universe)
                factor_df = (wide.rename_axis("trade_date")
                             .rename_axis("ts_code", axis=1)
                             .stack()
                             .rename("factor_value")
                             .reset_index())
                factor_df = factor_df[["ts_code", "trade_date", "factor_value"]]
            else:
                return None
        except Exception as e:
            print(f"    [IC Computer] {factor_name}: 向量化 eval 失败: "
                  f"{type(e).__name__}: {str(e)[:100]}")
            return None

        if factor_df is None or len(factor_df) == 0:
            return None
        factor_df = factor_df.dropna(subset=["factor_value"])
        factor_df = factor_df[np.isfinite(factor_df["factor_value"])]
        return factor_df

    def _get_wide_field(self, field: str) -> Optional[pd.DataFrame]:
        """按需将长表某字段 pivot 成宽表 (index=日期, columns=股票), 带缓存。"""
        if field in self._wide_cache:
            return self._wide_cache[field]
        if not self._loaded:
            self._load_data()
        if self._price_df is None or field not in self._price_df.columns:
            return None
        sub = self._price_df[["trade_date", "ts_code", field]].dropna(subset=[field])
        # 去重: merge 多数据源可能引入重复 (ts_code, trade_date), unstack 前必须唯一
        sub = sub.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        if len(sub) == 0:
            return None
        wide = sub.set_index(["trade_date", "ts_code"])[field].unstack()
        wide = wide.reindex(index=self._all_dates, columns=self._universe)
        self._wide_cache[field] = wide
        return wide

    # ── v0.5.2: S2 库内相关性计算 ──────────────────────────

    def compute_correlation(
        self, formula_a: str, formula_b: str,
        label_a: str = "A", label_b: str = "B",
    ) -> Optional[float]:
        """
        计算两个因子公式的横截面 Spearman 秩相关（均值）。

        对每个共同交易日，计算两个因子在 universe 上的横截面
        Spearman rank correlation，然后取所有日期的均值。

        Returns:
            float 相关系数均值 (0~1) 或 None（计算失败）
        """
        df_a = self._eval_formula(formula_a, label_a)
        df_b = self._eval_formula(formula_b, label_b)
        if df_a is None or df_b is None:
            return None

        merged = df_a.merge(df_b, on=["ts_code", "trade_date"],
                            suffixes=("_a", "_b"), how="inner")
        if len(merged) < 100:
            return None

        cors = []
        for date, group in merged.groupby("trade_date"):
            if len(group) < self.min_stocks_per_day:
                continue
            rk_a = group["factor_value_a"].rank()
            rk_b = group["factor_value_b"].rank()
            cor = rk_a.corr(rk_b)
            if not np.isnan(cor):
                cors.append(abs(cor))

        if len(cors) < 10:
            return None
        return round(float(np.mean(cors)), 4)

    def compute_max_corr_vs_library(
        self,
        candidate_formula: str,
        library_factors: List[Dict],
        candidate_name: str = "candidate",
    ) -> tuple:
        """
        计算候选因子与库中所有因子的最大相关性。

        Parameters
        ----------
        candidate_formula: 候选因子的公式字符串
        library_factors: 库因子列表 [{"factor_name": str, "formula": str}, ...]
        candidate_name: 候选名称（日志用）

        Returns
        -------
        (max_corr: float, max_corr_factor: str)
            - max_corr=0.0 表示库为空或无有效比较
            - max_corr=-1.0 表示候选公式评估失败
        """
        if not library_factors:
            return 0.0, ""

        # 先用范式过滤（快速缩减比较范围）
        # library_factors 可能包含 paradigm 字段
        # 如果没有则不预过滤
        max_corr = 0.0
        max_factor = ""
        n_compared = 0
        n_skipped = 0

        for lib in library_factors:
            lib_formula = lib.get("formula", lib.get("expression", ""))
            lib_name = lib.get("factor_name", lib.get("pattern_id", ""))
            if not lib_formula or not lib_name:
                n_skipped += 1
                continue
            if lib_name == candidate_name:
                continue  # 跳过自身

            cor = self.compute_correlation(
                candidate_formula, lib_formula,
                label_a=candidate_name, label_b=lib_name,
            )
            if cor is not None:
                n_compared += 1
                if cor > max_corr:
                    max_corr = cor
                    max_factor = lib_name

        if n_compared == 0:
            # 无法比较（可能公式格式不兼容）→ 返回 -1 哨兵值
            return -1.0, ""

        return max_corr, max_factor

    def compute_max_corr_vs_library_batch(
        self,
        candidate_formulas: List[str],
        library_factors: List[Dict],
        candidate_names: Optional[List[str]] = None,
        sample_stocks: int = 0,
    ) -> Dict[str, tuple]:
        """
        批量版 compute_max_corr_vs_library (v0.6.1 修复 S2 性能/稳定性):
        - 库模板只 eval 一次并缓存 (旧版每候选重算全部库模板 → NxM 次重复 eval)
        - eval 失败的坏模板预筛剔除, 不再对每个候选重复失败
        - 候选公式也只 eval 一次

        sample_stocks (2026-09-01 P1 转正配套): >0 时按固定种子 (42) 抽取
        sample_stocks 只股票做相关计算 (与 FactorICComputer IC 口径一致的
        120 只) — 多样性折扣是软引导信号, 不需要全市场精度;
        全市场 5763×1500天 groupby 每对 ~2s, 育种模板池 100+ 模板冷启动
        需 ~60min; 采样后 ~80s。默认 0 = 全市场 (S2 门禁等既有调用方不变)。

        Returns:
            {candidate_name: (max_corr, max_corr_factor)}
            - max_corr = -1.0 表示候选公式评估失败 (保守放行)
            - max_corr = 0.0 表示无有效比较
        """
        if candidate_names is None:
            candidate_names = [f"cand_{i}" for i in range(len(candidate_formulas))]
        out: Dict[str, tuple] = {}

        if not library_factors:
            for n in candidate_names:
                out[n] = (0.0, "")
            return out

        # 0) 采样股票集合 (固定种子, 跨库/候选一致)
        sample_codes: Optional[set] = None
        if sample_stocks and sample_stocks > 0:
            try:
                if not self._loaded:
                    self._load_data()
                rng = np.random.RandomState(42)
                codes = sorted(self._universe)
                if len(codes) > sample_stocks:
                    sample_codes = set(rng.choice(codes, size=sample_stocks,
                                                  replace=False))
                else:
                    sample_codes = set(codes)
            except Exception:
                sample_codes = None  # 采样失败退回全市场

        def _filter_df(df):
            if df is None or sample_codes is None:
                return df
            return df[df["ts_code"].isin(sample_codes)]

        # 1) 库模板探活 + 缓存 (坏模板一次性剔除)
        lib_cache: List[tuple] = []  # (lib_name, factor_df)
        for lib in library_factors:
            lib_formula = lib.get("formula", lib.get("expression", ""))
            lib_name = lib.get("factor_name", lib.get("pattern_id", ""))
            if not lib_formula or not lib_name:
                continue
            df = _filter_df(self._eval_formula(str(lib_formula), str(lib_name)))
            if df is None or len(df) == 0:
                continue  # 坏模板: _eval_formula 已打印失败日志, 此处静默剔除
            lib_cache.append((str(lib_name), df))
        if not lib_cache:
            for n in candidate_names:
                out[n] = (0.0, "")
            return out

        # 2) 候选逐次 eval + 与缓存库信号算相关
        for name, formula in zip(candidate_names, candidate_formulas):
            if not formula:
                out[name] = (0.0, "")
                continue
            if name in [ln for ln, _ in lib_cache]:
                out[name] = (0.0, "")  # 跳过自身
                continue
            cand_df = _filter_df(self._eval_formula(str(formula), str(name)))
            if cand_df is None or len(cand_df) == 0:
                out[name] = (-1.0, "")
                continue
            max_corr, max_factor, n_compared = 0.0, "", 0
            for lib_name, lib_df in lib_cache:
                try:
                    merged = cand_df.merge(
                        lib_df, on=["ts_code", "trade_date"], how="inner",
                        suffixes=("_c", "_l"))
                except Exception:
                    continue
                if len(merged) < 100:
                    continue
                cors = []
                for date, group in merged.groupby("trade_date"):
                    if len(group) < self.min_stocks_per_day:
                        continue
                    rk_c = group["factor_value_c"].rank()
                    rk_l = group["factor_value_l"].rank()
                    cor = rk_c.corr(rk_l)
                    if not np.isnan(cor):
                        cors.append(abs(cor))
                if len(cors) >= 10:
                    n_compared += 1
                    corr_mean = float(np.mean(cors))
                    if corr_mean > max_corr:
                        max_corr, max_factor = corr_mean, lib_name
            if n_compared == 0:
                out[name] = (-1.0, "")
            else:
                out[name] = (round(max_corr, 4), max_factor)

        return out

    # ── v0.5.2: S5 轻量级两年正向过滤 ──────────────────────

    def compute_yearly_ic(
        self, formula: str, factor_name: str = "unknown",
    ) -> Dict[str, Optional[Dict]]:
        """
        按年份拆分计算 IC，用于 S5 轻量级两年正向过滤。

        Returns:
            {
                "2025": {"ic": float, "icir": float, "n_days": int} or None,
                "2026": {"ic": float, "icir": float, "n_days": int} or None,
                "all":  {"ic": float, "icir": float, "n_days": int},
            }
        """
        if not self._loaded:
            self._load_data()

        if self._price_df is None or len(self._universe) == 0:
            return {"all": {"ic": 0, "icir": 0, "n_days": 0}}

        factor_df = self._eval_formula(formula, factor_name)
        if factor_df is None or len(factor_df) == 0:
            return {"all": {"ic": 0, "icir": 0, "n_days": 0}}

        price_data = self._price_df[self._price_df["ts_code"].isin(self._universe)][
            ["ts_code", "trade_date", "close"]
        ].drop_duplicates(subset=["ts_code", "trade_date"])

        merged = factor_df.merge(price_data, on=["ts_code", "trade_date"], how="inner")
        merged = merged.sort_values(["ts_code", "trade_date"])
        merged["fwd_return"] = (
            merged.groupby("ts_code")["close"].shift(-self.forward_period) / merged["close"] - 1
        )
        merged["year"] = merged["trade_date"].dt.year
        merged = merged.dropna(subset=["factor_value", "fwd_return"])

        result = {"all": self.compute(formula, factor_name)}
        for year in [2025, 2026]:
            year_data = merged[merged["year"] == year]
            ic_series = []
            for date, group in year_data.groupby("trade_date"):
                if len(group) < self.min_stocks_per_day:
                    continue
                rk_f = group["factor_value"].rank()
                rk_r = group["fwd_return"].rank()
                ic = rk_f.corr(rk_r)
                if not np.isnan(ic):
                    ic_series.append(ic)

            if len(ic_series) >= 5:
                ic_arr = np.array(ic_series)
                result[str(year)] = {
                    "ic": round(float(np.nanmean(ic_arr)), 6),
                    "icir": round(float(np.nanmean(ic_arr) / np.nanstd(ic_arr)), 4)
                    if np.nanstd(ic_arr) > 0 else 0.0,
                    "n_days": len(ic_series),
                }
            else:
                result[str(year)] = None

        return result

    def _normalize_to_pandas(self, formula: str) -> Optional[str]:
        """DSL 格式 → pandas infix 转换（从 S5JointFilter 移植）"""
        dsl_pattern = re.match(
            r'^(sub|add|mul|div|neg|abs|sqrt|log|sign|ma|ts_mean|ts_std|'
            r'ts_min|ts_max|ts_sum|ts_delta|ts_delay|delta|ema|roc|rank|'
            r'zscore|cs_zscore|demean|normalize|ts_corr|ts_cov|ts_rank|'
            r'ts_skewness|ts_kurtosis|ts_regression|ts_decay_linear)\(',
            formula
        )
        if not dsl_pattern:
            return formula  # 已是 pandas infix

        try:
            from factor_expression_tree import dsl_to_pandas_infix
            converted = dsl_to_pandas_infix(formula)
            if converted:
                return converted
        except ImportError:
            pass
        return formula


# ── 便捷函数 ──────────────────────────────────────────────

def compute_rank_ic_standalone(
    formula: str,
    factor_name: str = "unknown",
    price_path: Path = None,
) -> Dict[str, float]:
    """
    单次调用: 从 daily_prices.csv 计算因子 Rank IC。
    不依赖 S5JointFilter。
    """
    comp = FactorICComputer(price_path=price_path)
    return comp.compute(formula, factor_name)


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    comp = FactorICComputer()

    formulas = [
        "(close - close.rolling(20).mean()) / close.rolling(20).std()",
        "-(close.pct_change(5).rolling(20).mean())",
        "rank(div(ts_mean(volume, 10), ts_mean(volume, 60)))",
    ]

    for f in formulas:
        result = comp.compute(f, "test")
        print(f"  {f[:60]}...")
        print(f"    → IC={result['ic']:.4f}, ICIR={result['icir']:.4f}, n_days={result['n_days']}")
