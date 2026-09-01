# -*- coding: utf-8 -*-
"""
Diversity Discount — Alpha2 MaxCorr 折扣移植 (P-20260831, P1)
================================================================================
背景 (research/alpha2_vs_fuxi_nvwa_20260831.md §P1):
  Alpha2 把多样性写进 reward: Perf = (1 - MaxCorr(z, G)) * IC(z)。
  伏羲的多样性全部在出口层 (S2 库内相关硬门槛 / MMR / Originality) —
  "挖了再筛 = 浪费搜索预算"。本模块把折扣前移到进化选择层:

    1. gp_breed 父本分层采样 (GPBreeder._layered_template_score):
       与"已 JQ 验证"因子高相关的模板降采样 → 变异往未探索方向走。
    2. 主线 _phase_evaluate S2 之后: jq_max_corr 影子字段落盘
       (观察哪些候选在复制已验证信号, 为未来 enforce 积累数据)。

参照集铁律 (防 926a 式方向翻转传播):
  只参照【JQ 正面验证】的因子 (JQ_PASSED / JQ_MARGINAL / PASS) —
  JQ 实测为负/无效的因子 (JQ_FAILED / WEAK / JQ_WEAK_NEGATIVE) 不进参照集:
  用不可信方向做折扣会把错误方向传播进进化选择。
  相关性取 |corr| (逐日截面 rank corr 绝对值均值) — 正负相关都算"信号方向被占用"。

模式 (与 P0 维度预剪枝同一 SOP):
  FUXI_DIV_DISCOUNT = off     → 完全关闭 (零开销)
                      shadow  → 计算 + 记录统计, 不改任何行为 (默认)
                      enforce → 折扣生效 (需冠军校准验收后转正)

折扣公式:
  score' = score * max(FLOOR, 1 - w * jq_max_corr)
  默认 w=0.10 (对比报告建议 0.05~0.15 中值), FLOOR=0.5 (软引导, 最多砍半)。
"""

import os
import re
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 项目根 = 本文件上三级 (forge -> factor_alchemy -> research -> quant)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── 可调参数 ──
DIV_DISCOUNT_W = 0.10          # 折扣权重 (软引导)
DIV_DISCOUNT_FLOOR = 0.5       # 折扣下限: 分数最多砍到 50%
_MIN_PAIRS = 100               # 相关计算最少样本 (与 FactorICComputer 一致)
_POSITIVE_OUTCOMES = {"PASS", "JQ_PASSED", "JQ_MARGINAL"}   # 正参照白名单
_POSITIVE_JQ_OUTCOMES = {"JQ_PASSED", "JQ_MARGINAL"}        # lane_calibration 白名单

# ── 统计 (影子模式观测窗口) ──
class _Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.n_queries = 0          # 折扣查询次数
        self.n_corr_computed = 0    # 成功算出 corr 的查询
        self.n_eval_fail = 0        # 公式评估失败 (保守 corr=0)
        self.n_self_hit = 0         # 模板与参照因子同名 (corr=1.0)
        self.corr_bucket = {  # corr 分布桶 (折扣强度分布)
            "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}

    def record(self, corr: float, computed: bool, eval_fail: bool = False,
               self_hit: bool = False):
        with self.lock:
            self.n_queries += 1
            if self_hit:
                self.n_self_hit += 1
            elif eval_fail:
                self.n_eval_fail += 1
            elif computed:
                self.n_corr_computed += 1
                if corr < 0.2: self.corr_bucket["0.0-0.2"] += 1
                elif corr < 0.4: self.corr_bucket["0.2-0.4"] += 1
                elif corr < 0.6: self.corr_bucket["0.4-0.6"] += 1
                elif corr < 0.8: self.corr_bucket["0.6-0.8"] += 1
                else: self.corr_bucket["0.8-1.0"] += 1

    def snapshot(self) -> Dict:
        with self.lock:
            return {"n_queries": self.n_queries,
                    "n_corr_computed": self.n_corr_computed,
                    "n_eval_fail": self.n_eval_fail,
                    "n_self_hit": self.n_self_hit,
                    "corr_bucket": dict(self.corr_bucket)}

    def reset(self):
        with self.lock:
            self.n_queries = self.n_corr_computed = 0
            self.n_eval_fail = self.n_self_hit = 0
            self.corr_bucket = {k: 0 for k in self.corr_bucket}


COUNTERS = _Counters()


# ═══════════════════════════════════════════════════════════
# 1. 模式
# ═══════════════════════════════════════════════════════════

def get_mode() -> str:
    """off / shadow / enforce (默认 shadow, 生产零行为变化)"""
    m = os.environ.get("FUXI_DIV_DISCOUNT", "shadow").strip().lower()
    return m if m in ("off", "shadow", "enforce") else "shadow"


def is_enforce() -> bool:
    return get_mode() == "enforce"


# ═══════════════════════════════════════════════════════════
# 2. JQ 正参照库加载
# ═══════════════════════════════════════════════════════════

def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def load_jq_reference_library() -> List[Dict]:
    """加载 JQ 正面验证参照集 (pandas 公式, 按公式去重)。

    源1: data/experience_memory.json — attempts 中 jq_verified=True
         且 outcome ∈ {PASS, JQ_PASSED, JQ_MARGINAL} (34 个 JQ 验证中约 11 个正面)。
    源2: data/lane_calibration.json — points 中 jq_outcome ∈ {JQ_PASSED, JQ_MARGINAL}
          (D+ 闭环滚动维护的周频校准锚点)。

    返回: [{factor_name, formula}, ...]
    """
    lib: List[Dict] = []
    seen_formulas = set()

    def _add(name: str, formula: str):
        if not formula:
            return
        key = _norm(formula)
        if not key or key in seen_formulas:
            return
        seen_formulas.add(key)
        lib.append({"factor_name": str(name)[:80], "formula": str(formula)})

    # 源1: 经验记忆
    try:
        em_path = _PROJECT_ROOT / "data" / "experience_memory.json"
        if em_path.exists():
            with open(em_path, "r", encoding="utf-8") as f:
                em = json.load(f)
            for a in em.get("attempts", []):
                if not isinstance(a, dict):
                    continue
                if not a.get("jq_verified"):
                    continue
                if str(a.get("outcome", "")) not in _POSITIVE_OUTCOMES:
                    continue
                fml = a.get("formula", a.get("expression", ""))
                _add(a.get("factor_name", a.get("name", "jq_ref")), fml)
    except Exception as e:
        print(f"  [DivDisc] ⚠️ experience_memory 加载失败: {e}")

    # 源2: lane_calibration 周频锚点
    try:
        lc_path = _PROJECT_ROOT / "data" / "lane_calibration.json"
        if lc_path.exists():
            with open(lc_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            for p in cal.get("points", []):
                if not isinstance(p, dict):
                    continue
                if str(p.get("jq_outcome", "")) not in _POSITIVE_JQ_OUTCOMES:
                    continue
                _add(p.get("name", "jq_anchor"), p.get("formula", ""))
    except Exception as e:
        print(f"  [DivDisc] ⚠️ lane_calibration 加载失败: {e}")

    return lib


# ═══════════════════════════════════════════════════════════
# 3. JQ 相关缓存 (模板公式 → jq_max_corr)
# ═══════════════════════════════════════════════════════════

class JQDiversityCache:
    """惰性批量计算 模板公式 vs JQ 正参照库 的最大 |rank corr|。

    复用 FactorICComputer.compute_max_corr_vs_library_batch:
      - 库模板只 eval 一次; 候选公式 eval 一次; 坏公式预筛。
      - max_corr = 逐日截面 rank corr 绝对值均值 (天然防方向翻转)。
    缓存按归一化公式字符串 key, 跨轮次复用 (GPBreeder 实例存活期)。

    磁盘持久化 (2026-09-01 转正配套): data/jq_div_corr_cache.json,
      键 = norm(formula), 值 = corr; lib_sig = 参照库名+公式哈希 —
      参照库变化 (D+ 闭环滚动) 时整库失效重算, 模板 corr 跨轮次复用,
      避免每轮育种对模板池重算 (~40s/模板 → 首轮后近零)。

    注意: compute_max_corr_vs_library_batch 会跳过"候选名 == 库名"的自身比较
    (返回 0.0) — 但那恰恰说明该模板就是 JQ 验证因子本身, corr 应为 1.0,
    在此处修正为 1.0 (该方向 100% 已被占用)。
    """

    _CACHE_FILE = _PROJECT_ROOT / "data" / "jq_div_corr_cache.json"
    _SAVE_INTERVAL_S = 60.0   # 磁盘回写节流 (避免每查询都写盘)

    def __init__(self, ic_comp=None, batch_size: int = 64):
        self._ic_comp = ic_comp
        self._batch_size = batch_size
        self._cache: Dict[str, float] = {}   # norm(formula) -> jq_max_corr
        self._lib: Optional[List[Dict]] = None
        self._lib_names: Optional[set] = None
        self._lib_sig: str = ""
        self._lock = threading.Lock()
        self._lib_loaded = False
        self._dirty = False
        self._last_save = 0.0

    def _load_disk_cache(self):
        """参照库签名匹配时载入磁盘缓存 (异常静默, 缓存缺失即冷启动)。"""
        try:
            if not self._CACHE_FILE.exists():
                return
            with open(self._CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            if data.get("lib_sig") != self._lib_sig:
                return  # 参照库已变化 → 整库失效
            entries = data.get("entries")
            if isinstance(entries, dict):
                for k, v in entries.items():
                    if isinstance(v, (int, float)) and k:
                        self._cache[str(k)] = float(v)
        except Exception:
            return

    def _save_disk_cache(self, force: bool = False):
        """磁盘回写 (节流 + 异常静默, 失败不影响主流程)。

        force=True: 批量查询收尾强制落盘 — 修复 2026-09-01 首轮验证发现的
        "节流吞最终回写"问题 (批量后 ~68 条只落 2 条, 进程退出即丢)。
        """
        try:
            now = time.time()
            if not self._dirty or (not force and (now - self._last_save) < self._SAVE_INTERVAL_S):
                return
            with self._lock:
                if not self._dirty:
                    return
                payload = {"lib_sig": self._lib_sig,
                           "entries": dict(self._cache),
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                self._dirty = False
                self._last_save = now
            tmp = self._CACHE_FILE.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self._CACHE_FILE)
        except Exception:
            pass

    def flush(self):
        """强制落盘 (批量查询收尾 / 进程退出时调用)。"""
        self._save_disk_cache(force=True)

    def _ensure_lib(self):
        if not self._lib_loaded:
            self._lib = load_jq_reference_library()
            self._lib_names = {f["factor_name"] for f in self._lib}
            self._lib_sig = hashlib.md5(
                "|".join(f"{f['factor_name']}::{_norm(f['formula'])}"
                         for f in sorted(self._lib, key=lambda x: x["factor_name"]))
                .encode("utf-8")).hexdigest()
            self._load_disk_cache()
            self._lib_loaded = True

    def _ensure_ic_comp(self):
        if self._ic_comp is None:
            from factor_ic_computer import FactorICComputer
            self._ic_comp = FactorICComputer()  # 全市场加载, 首次慢

    def reference_size(self) -> int:
        self._ensure_lib()
        return len(self._lib)

    def _cache_set(self, key: str, v: float):
        with self._lock:
            self._cache[key] = v
            self._dirty = True
        self._save_disk_cache()

    def get_corr(self, formula: str, name: str = "") -> float:
        """单公式查询 (带内存+磁盘缓存)。评估失败 → 0.0 (保守: 不知道就不折扣)。"""
        key = _norm(formula)
        if not key:
            COUNTERS.record(0.0, computed=False)
            return 0.0
        with self._lock:
            if key in self._cache:
                v = self._cache[key]
                COUNTERS.record(v, computed=True)
                return v
        v = self._compute_batch([formula], [name or "q"]).get(name or "q", 0.0)
        self._cache_set(key, v)
        return v

    def get_corrs_batch(self, formulas: List[str],
                        names: List[str]) -> Dict[str, float]:
        """批量查询 (未命中缓存的部分一次性算)。返回 {name: corr}"""
        out: Dict[str, float] = {}
        pending_f, pending_n = [], []
        for f, n in zip(formulas, names):
            key = _norm(f)
            if not key:
                out[n] = 0.0
                continue
            with self._lock:
                if key in self._cache:
                    out[n] = self._cache[key]
                    continue
            pending_f.append(f)
            pending_n.append(n)
        if pending_f:
            res = self._compute_batch(pending_f, pending_n)
            for f, n in zip(pending_f, pending_n):
                v = res.get(n, 0.0)
                out[n] = v
                self._cache_set(_norm(f), v)
            self.flush()  # 批量收尾强制落盘 (节流会吞批量内快速连写)
        return out

    def _compute_batch(self, formulas: List[str],
                       names: List[str]) -> Dict[str, float]:
        """调用 FactorICComputer 批量计算, 修正 self-skip 语义。

        采样口径 (2026-09-01 转正配套): sample_stocks=120 固定种子 —
        多样性折扣是软引导信号, 不需要全市场精度; 全市场口径每对 ~2s,
        育种模板池冷启动 ~60min, 采样后 ~80s (与 FactorICComputer IC
        口径一致的 120 只)。
        """
        self._ensure_lib()
        if not self._lib:
            for n in names:
                COUNTERS.record(0.0, computed=False)
            return {n: 0.0 for n in names}
        self._ensure_ic_comp()
        try:
            raw = self._ic_comp.compute_max_corr_vs_library_batch(
                formulas, self._lib, names, sample_stocks=120)
        except Exception as e:
            print(f"  [DivDisc] ⚠️ 批量相关计算失败 (保守 corr=0): {e}")
            for n in names:
                COUNTERS.record(0.0, computed=False, eval_fail=True)
            return {n: 0.0 for n in names}

        out: Dict[str, float] = {}
        for n, f in zip(names, formulas):
            val, max_factor = raw.get(n, (0.0, ""))
            if val < 0:  # 候选评估失败
                COUNTERS.record(0.0, computed=False, eval_fail=True)
                out[n] = 0.0
                continue
            if n in (self._lib_names or set()) or (
                    max_factor and _norm(max_factor) == _norm(f)):
                # 候选就是参照因子本身 → corr=1.0 (方向 100% 占用)
                COUNTERS.record(1.0, computed=False, self_hit=True)
                out[n] = 1.0
                continue
            v = float(min(max(val, 0.0), 1.0))
            COUNTERS.record(v, computed=True)
            out[n] = v
        return out


# 进程内单例 (gp_breed 与主线共享一个 FactorICComputer, 避免重复加载数据)
_GLOBAL_CACHE: Optional[JQDiversityCache] = None
_GLOBAL_LOCK = threading.Lock()


def get_shared_cache(ic_comp=None) -> JQDiversityCache:
    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = JQDiversityCache(ic_comp=ic_comp)
        elif ic_comp is not None and _GLOBAL_CACHE._ic_comp is None:
            _GLOBAL_CACHE._ic_comp = ic_comp  # 注入已加载的共享实例
        return _GLOBAL_CACHE


def _flush_global_cache_on_exit():
    """进程退出兜底落盘 (S2-post 单查询路径的收尾保障)。"""
    try:
        if _GLOBAL_CACHE is not None:
            _GLOBAL_CACHE.flush()
    except Exception:
        pass


import atexit as _atexit
_atexit.register(_flush_global_cache_on_exit)


def reset_shared_cache():
    global _GLOBAL_CACHE
    with _GLOBAL_LOCK:
        _GLOBAL_CACHE = None


# ═══════════════════════════════════════════════════════════
# 4. 折扣应用
# ═══════════════════════════════════════════════════════════

def apply_discount(score: float, jq_max_corr: float,
                   w: float = DIV_DISCOUNT_W,
                   floor: float = DIV_DISCOUNT_FLOOR) -> float:
    """score' = score * max(floor, 1 - w * jq_max_corr)。enforce 模式专用。"""
    if w <= 0:
        return score
    disc = max(floor, 1.0 - w * max(min(float(jq_max_corr), 1.0), 0.0))
    return float(score) * disc


def discount_factor(jq_max_corr: float, w: float = DIV_DISCOUNT_W,
                    floor: float = DIV_DISCOUNT_FLOOR) -> float:
    """返回折扣系数 (影子统计用, 不改 score)"""
    if w <= 0:
        return 1.0
    return max(floor, 1.0 - w * max(min(float(jq_max_corr), 1.0), 0.0))
