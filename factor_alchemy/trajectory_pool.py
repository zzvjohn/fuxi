# -*- coding: utf-8 -*-
"""
P-20260819-005: 轨迹级进化父本池 + 冗余感知融合 (AlphaSeek 对齐)
====================================================================
AlphaSeek (arXiv 2608.13913, 2026-08-14) 核心思想: 进化单元应是
"完整研究轨迹" (假设→构建→验证→回测→反馈), 而非单条因子公式。
本模块把该思想映射到伏羲 R 阶段:

  1. load_success_trajectories   从 Trajectory Logger 提取成功轨迹
  2. build_parent_pool           轨迹 → 父本池 (按命中率加权 + 冷却)
  3. layer_sample_parents        按轨迹最强阶段 (hypothesis/expression/
                                 code/backtest) 分层采样父本
  4. redundancy_check            新因子与 SOTA 库 AST 结构冗余检测 (>0.7
                                 标记融合, 抑制 Red Sea 拥挤)

工程边界: 不改 MAB/三引擎/铁律; 由 RalphLoop 开关控制 (默认关),
S5 命中率下降即回滚。

创建: 2026-08-19 (P-20260819-005 落地)
"""
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ── 开关 (提案 rollback_plan: 默认关; 2026-08-19 用户指令开启试跑) ──
ENABLE_TRAJECTORY_POOL = True

# 成功轨迹判定阈值
SUCCESS_ICIR_MIN = 0.30          # 本地 S5 通过的 ICIR 下限
JQ_RETURN_MIN = 20.0             # JQ 回测收益下限 (百分比格式)

# 轨迹阶段 (AlphaSeek 四段式)
TRAJECTORY_PHASES = ["hypothesis", "expression", "code", "backtest"]

# 冷却天数: 同一条轨迹 30 天内不重复作父本
COOLDOWN_DAYS = 30

# 冗余融合阈值: AST 结构 Jaccard 相似度上限
REDUNDANCY_THRESHOLD = 0.7


def _data_dir() -> Path:
    """data 目录 = 本模块上级的上级 / data"""
    return Path(__file__).resolve().parent.parent.parent / "data"


# ═══════════════════════════════════════════════════════════════
# 1. 成功轨迹提取
# ═══════════════════════════════════════════════════════════════

def load_success_trajectories(data_dir: Optional[Path] = None) -> List[Dict]:
    """从 data/trajectory_log.json 提取成功轨迹。

    成功判定 (满足任一):
      - jq_validated 且 jq_return >= JQ_RETURN_MIN (JQ 真值验证正向)
      - final_outcome in ('PASSED', 'PASS')
      - final_icir >= SUCCESS_ICIR_MIN (本地 S5 达标)

    返回 List[dict]: 每项含 factor_name/formula/paradigm/score/
    best_step/hit 信息, 按综合分降序。
    """
    data_dir = data_dir or _data_dir()
    tlog_path = data_dir / "trajectory_log.json"
    if not tlog_path.exists():
        return []
    try:
        t = json.load(open(tlog_path, encoding="utf-8"))
    except Exception:
        return []
    trajs = t.get("trajectories", t) if isinstance(t, dict) else t
    if not isinstance(trajs, list):
        return []

    success = []
    for x in trajs:
        if not isinstance(x, dict):
            continue
        jq_ok = bool(x.get("jq_validated")) and (x.get("jq_return") or 0) >= JQ_RETURN_MIN
        outcome_ok = str(x.get("final_outcome", "")).upper() in ("PASSED", "PASS")
        icir_ok = (x.get("final_icir") or 0) >= SUCCESS_ICIR_MIN
        if not (jq_ok or outcome_ok or icir_ok):
            continue

        # 公式来源: expression 阶段 content 最可靠
        formula = ""
        for phase in ("expression", "code"):
            p = x.get(phase)
            if isinstance(p, dict) and p.get("content"):
                formula = str(p["content"]).strip()
                break
        if not formula and x.get("seed_factor"):
            formula = str(x["seed_factor"]).strip()
        if not formula:
            continue

        # 各阶段分数与最强阶段
        phase_scores = {}
        for pn in TRAJECTORY_PHASES:
            p = x.get(pn)
            if isinstance(p, dict) and isinstance(p.get("score"), (int, float)):
                phase_scores[pn] = float(p["score"])
        best_step = max(phase_scores, key=phase_scores.get) if phase_scores else "expression"
        score = x.get("overall_score") or 0
        if not score and phase_scores:
            score = sum(phase_scores.values()) / len(phase_scores)

        success.append({
            "factor_name": str(x.get("seed_factor") or x.get("trajectory_id") or "")[:60],
            "formula": formula,
            "paradigm": str(x.get("paradigm") or ""),
            "score": float(score),
            "final_icir": float(x.get("final_icir") or 0),
            "jq_return": float(x.get("jq_return") or 0),
            "jq_validated": bool(x.get("jq_validated")),
            "best_step": best_step,
            "phase_scores": phase_scores,
            "trajectory_id": x.get("trajectory_id", ""),
        })

    # JQ 验证过的轨迹权重更高 (JQ 是唯一真相源)
    for s in success:
        s["weight"] = s["score"] + (0.5 if s["jq_validated"] else 0.0)
    success.sort(key=lambda s: -s["weight"])
    return success


# ═══════════════════════════════════════════════════════════════
# 2. 父本池构建 + 分层采样
# ═══════════════════════════════════════════════════════════════

def _pool_state_path(data_dir: Path) -> Path:
    return data_dir / "trajectory_pool_state.json"


def _load_pool_state(data_dir: Path) -> Dict:
    p = _pool_state_path(data_dir)
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return {"used": {}}


def _save_pool_state(data_dir: Path, state: Dict) -> None:
    json.dump(state, open(_pool_state_path(data_dir), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def build_parent_pool(data_dir: Optional[Path] = None,
                      cooldown_days: int = COOLDOWN_DAYS,
                      max_pool: int = 20) -> List[Dict]:
    """成功轨迹 → 父本池 (公式去重 + 30 天冷却 + 上限裁剪)。

    去重: 归一化公式 (去空白/去数字常量) 相同者只保留综合分最高一条。
    冷却: trajectory_pool_state.json 记录上次使用日, 冷却期内跳过。
    """
    data_dir = data_dir or _data_dir()
    trajs = load_success_trajectories(data_dir)
    if not trajs:
        return []

    state = _load_pool_state(data_dir)
    used = state.get("used", {})
    now = datetime.now()
    cutoff = (now - timedelta(days=cooldown_days)).isoformat()[:10]

    seen = set()
    pool = []
    for t in trajs:
        norm = _normalize_formula(t["formula"])
        if norm in seen:
            continue
        key = t["trajectory_id"] or t["factor_name"] or norm[:40]
        if used.get(key, "") >= cutoff:
            continue  # 冷却期内
        seen.add(norm)
        t["_norm"] = norm
        t["_key"] = key
        pool.append(t)
        if len(pool) >= max_pool:
            break
    return pool


def mark_parent_used(data_dir: Optional[Path] = None,
                     parent_keys: Optional[List[str]] = None) -> None:
    """父本使用后记录冷却时间戳 (幂等)。"""
    if not parent_keys:
        return
    data_dir = data_dir or _data_dir()
    state = _load_pool_state(data_dir)
    used = state.setdefault("used", {})
    today = datetime.now().isoformat()[:10]
    for k in parent_keys:
        used[k] = today
    _save_pool_state(data_dir, state)


def layer_sample_parents(pool: List[Dict], n: int,
                         paradigm: Optional[str] = None) -> List[Dict]:
    """按轨迹最强阶段 (假设/公式/代码/回测) 分层采样父本。

    AlphaSeek 对齐: 变异算子按轨迹阶段分层, 此处等价实现为
    "父本按最强阶段分层均衡采样" — 保证每一层都有代表进入育种,
    避免单点冠军因子主导 (多样性提升)。
    """
    if not pool or n <= 0:
        return []
    if paradigm:
        same = [p for p in pool if p["paradigm"] == paradigm]
        pool = same + [p for p in pool if p["paradigm"] != paradigm]

    layers = {p: [] for p in TRAJECTORY_PHASES}
    for p in pool:
        layers.setdefault(p.get("best_step", "expression"), []).append(p)

    selected, seen = [], set()
    # 轮转分配: 每轮从每层各取 1 条 (层内按权重取最高)
    layer_names = [pn for pn in TRAJECTORY_PHASES if layers.get(pn)]
    round_robin = True
    while len(selected) < n and round_robin:
        round_robin = False
        for pn in layer_names:
            for p in layers[pn]:
                if p["_key"] in seen:
                    continue
                selected.append(p)
                seen.add(p["_key"])
                round_robin = True
                break
            if len(selected) >= n:
                break
    # 不足则按权重补
    if len(selected) < n:
        for p in pool:
            if p["_key"] in seen:
                continue
            selected.append(p)
            seen.add(p["_key"])
            if len(selected) >= n:
                break
    return selected[:n]


# ═══════════════════════════════════════════════════════════════
# 3. 冗余感知融合检测
# ═══════════════════════════════════════════════════════════════

def _normalize_formula(formula: str) -> str:
    """归一化: 去空白 + 数字常量替换为 <N> (结构指纹前处理)。"""
    f = re.sub(r"\s+", "", str(formula))
    f = re.sub(r"\d+\.?\d*", "<N>", f)
    return f


def ast_fingerprint(formula: str) -> Counter:
    """表达式树节点类型 multiset 指纹 (结构相似度用)。"""
    try:
        from factor_expression_tree import FactorExpressionParser
        parser = FactorExpressionParser()
        tree = parser.parse(formula)
        counter = Counter()
        for node in tree.get_all_subtrees() if hasattr(tree, "get_all_subtrees") else []:
            counter[node.op if hasattr(node, "op") else type(node).__name__] += 1
        if not counter:
            counter["root"] = 1
        return counter
    except Exception:
        # 解析失败降级: 字符 bigram 指纹
        f = _normalize_formula(formula)
        return Counter(f[i:i + 2] for i in range(max(1, len(f) - 1)))


def formula_similarity(f1: str, f2: str) -> float:
    """两条公式的结构相似度 = AST 指纹 Jaccard。"""
    if not f1 or not f2:
        return 0.0
    if _normalize_formula(f1) == _normalize_formula(f2):
        return 1.0
    c1, c2 = ast_fingerprint(f1), ast_fingerprint(f2)
    if not c1 and not c2:
        return 0.0
    inter = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    return inter / union if union else 0.0


def redundancy_check(candidate_formula: str,
                     sota_formulas: List[str],
                     threshold: float = REDUNDANCY_THRESHOLD) -> Dict:
    """新因子与 SOTA 库冗余检测。

    Args:
        candidate_formula: 新候选公式
        sota_formulas: SOTA 库公式列表 (success_templates + 近期 attempts)
        threshold: 相似度阈值 (默认 0.7)

    Returns:
        {"redundant": bool, "max_sim": float, "matched": str}
        redundant=True 时调用方应触发融合而非重复入库。
    """
    best, best_f = 0.0, ""
    for f in sota_formulas:
        if not f:
            continue
        sim = formula_similarity(candidate_formula, str(f))
        if sim > best:
            best, best_f = sim, str(f)[:80]
        if best >= threshold:
            break
    return {"redundant": best >= threshold, "max_sim": round(best, 3), "matched": best_f}


def collect_sota_formulas(memory) -> List[str]:
    """从 Memory 收集 SOTA 库公式 (success_templates + 近期 attempts)。"""
    formulas = []
    try:
        for t in getattr(memory, "success_templates", []) or []:
            f = getattr(t, "formula", None) or (t.get("formula") if isinstance(t, dict) else None)
            if f:
                formulas.append(str(f))
    except Exception:
        pass
    try:
        for a in getattr(memory, "attempts", []) or []:
            if isinstance(a, dict) and a.get("formula"):
                formulas.append(str(a["formula"]))
    except Exception:
        pass
    return formulas


# ═══════════════════════════════════════════════════════════════
# 4. 一键入口 (RalphLoop 调用)
# ═══════════════════════════════════════════════════════════════

def get_trajectory_parents(data_dir: Optional[Path] = None,
                           n: int = 5,
                           paradigm: Optional[str] = None) -> List[Dict]:
    """一键入口: 构建父本池 + 分层采样 n 条 + 标记冷却。

    返回 [{factor_name, formula, paradigm, best_step, weight}] —
    可直接注入 gp_breed 种子/父本列表。
    """
    if not ENABLE_TRAJECTORY_POOL:
        return []
    data_dir = data_dir or _data_dir()
    pool = build_parent_pool(data_dir)
    if not pool:
        return []
    sampled = layer_sample_parents(pool, n, paradigm=paradigm)
    mark_parent_used(data_dir, [p["_key"] for p in sampled])
    return [{k: v for k, v in p.items() if k in
             ("factor_name", "formula", "paradigm", "best_step", "weight", "final_icir")}
            for p in sampled]
