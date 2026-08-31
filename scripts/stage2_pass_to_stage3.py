"""
Stage 2 → Stage 3 桥接: 自动将 Stage 2 本地 PASS 因子注入因子库和 Experience Memory
=====================================================================================

读取 stage1_factor_proposals.json 中 status=PASS 的因子，
自动注入到:
  1. unified_factor_pool.csv (因子库)
  2. experience_memory.json (Memory attempts + success_templates)

调用方式:
    python scripts/stage2_pass_to_stage3.py                     # 注入所有未入库的 PASS 因子
    python scripts/stage2_pass_to_stage3.py --dry-run           # 预览模式，不实际写入
    python scripts/stage2_pass_to_stage3.py --factor NAME       # 仅注入指定因子

设计目标:
  - 消除 Stage 2 PASS → Stage 3 育种池 之间的人工手动桥接
  - 每次 Stage 2 产出 PASS 因子后自动调用
  - 去重保护: 已存在于因子库/Memory 的因子不会重复注入
"""

import json
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROPOSALS_FILE = DATA_DIR / 'stage1_factor_proposals.json'
POOL_CSV = DATA_DIR / 'unified_factor_pool.csv'
MEMORY_FILE = DATA_DIR / 'experience_memory.json'


def load_existing_pool_names() -> set:
    """从 unified_factor_pool.csv 读取已存在的因子名。"""
    if not POOL_CSV.exists():
        return set()
    with open(POOL_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return {row.get('name', '') for row in reader}


def load_existing_memory_names() -> set:
    """从 experience_memory.json 读取已有记录和模板的因子名。"""
    if not MEMORY_FILE.exists():
        return set(), set()
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        mem = json.load(f)
    attempt_names = {a.get('factor_name', '') for a in mem.get('attempts', [])}
    template_names = {t.get('pattern_id', '') for t in mem.get('success_templates', [])}
    return attempt_names, template_names


def load_pass_factors() -> list:
    """从 stage1_factor_proposals.json 读取 status=PASS 的因子。"""
    if not PROPOSALS_FILE.exists():
        print(f"[WARN] 未找到 {PROPOSALS_FILE}")
        return []

    with open(PROPOSALS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    proposals = data.get('proposals', [])
    pass_factors = []

    for p in proposals:
        name = p.get('factor_name', '')
        status = p.get('status', '')

        # 多种方式判断 PASS
        is_pass = (
            status.upper() == 'PASS'
            or p.get('test_result', '').upper() == 'PASS'
            or float(p.get('icir', 0) or 0) >= 0.3  # ICIR >= 0.3 的因子也考虑
        )

        if not is_pass:
            continue
        if not name:
            continue

        formula = p.get('formula_pandas', p.get('formula', ''))
        if not formula:
            continue

        pass_factors.append(p)

    return pass_factors


def inject_to_pool(factor: dict, pool_names: set, fieldnames: list) -> dict | None:
    """将因子注入 CSV 因子库，返回写入的行数据。"""
    name = factor.get('factor_name', '')
    if name in pool_names:
        return None  # 已存在

    formula = factor.get('formula_pandas', factor.get('formula', ''))
    paradigm_map = {
        'attention': '情绪×日内',
        'flow_microstructure': '资金流',
        'momentum': '动量反转',
        'mean_reversion': '均值回复',
        'liquidity': '流动性',
        'volatility': '波动率',
        'sector_rotation': '行业轮动',
    }
    paradigm = paradigm_map.get(
        factor.get('paradigm', ''),
        factor.get('paradigm', '未分类')
    )

    icir_val = float(factor.get('icir', 0) or 0)
    plus_ic = factor.get('plus_ic_pct', factor.get('+ic%', ''))
    if not plus_ic:
        plus_ic = f"{float(factor.get('plus_ic', 0) or icir_val * 100):.1f}" if icir_val else ''

    row = {fn: '' for fn in fieldnames}
    row.update({
        'name': name,
        'label': factor.get('label', name),
        'round': 'daily',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'daily_icir': str(round(icir_val, 4)),
        '+ic_pct': str(plus_ic),
        'category': factor.get('paradigm', ''),
        'status': 'pass',
        'icir': str(round(icir_val, 4)),
        'hypothesis': factor.get('economic_rationale', '')[:300],
        'logic': factor.get('economic_rationale', '')[:200],
        'formula': formula[:500],
        'direction': factor.get('direction', 'long'),
        'source': 'stage2_pass',
        'exploration_basis': factor.get('source_proposal', ''),
        'paradigm': paradigm,
        'operators': _extract_operators(formula),
        'windows': '',
    })
    return row


def inject_to_memory(factor: dict, attempt_names: set, template_names: set, mem_data: dict):
    """将因子注入 Experience Memory (attempts + success_templates)。"""
    name = factor.get('factor_name', '')
    formula = factor.get('formula_pandas', factor.get('formula', ''))
    paradigm = factor.get('paradigm', '未分类')
    icir_val = float(factor.get('icir', 0) or 0)

    # 注入 attempts
    if name not in attempt_names:
        attempt_id = f"mem_{len(mem_data.get('attempts', [])):04d}"
        record = {
            "id": attempt_id,
            "factor_name": name,
            "formula": formula[:500],
            "paradigm": paradigm,
            "category": factor.get('paradigm', ''),
            "outcome": "PASS",
            "fri": {"score": 0.5, "grade": "B", "precision": 0.0,
                     "persistence": 0.0, "consistency": 0.0, "novelty": 0.0},
            "icir": round(icir_val, 4),
            "psi": {"r_squared": 0.0, "independence": 1.0, "max_corr_factor": ""},
            "tags": ["stage2_pass", paradigm],
            "lessons": [factor.get('economic_rationale', '')[:200]],
            "timestamp": datetime.now().isoformat(),
        }
        mem_data.setdefault('attempts', []).append(record)
        # 更新统计
        stats = mem_data.setdefault('stats', {})
        stats['total_attempts'] = stats.get('total_attempts', 0) + 1
        stats['total_pass'] = stats.get('total_pass', 0) + 1

    # v0.8: 模板池准入门槛收紧 — Stage2 PASS 不再直接写 success_templates。
    # 模板池 (GP 亲本/LLM 约束来源) 只接受:
    #   1. Ralph Loop S5 通过路径 (_upsert_success_pattern, level=s5_passed)
    #   2. JQ 单因子验证通过路径 (form_from_jq, level=jq_single)
    #   3. champion 组合级模板 (迁移时标 jq_composite)
    # Stage2 PASS 的经验价值由 attempts 完整保留 (参与 motif 统计/LLM prompt)。


def _extract_operators(formula: str) -> str:
    """从 pandas 公式中提取操作符关键词。"""
    ops = []
    keywords = ['pct_change', 'rolling', 'std', 'mean', 'diff', 'rank',
                'shift', 'corr', 'skew', 'kurt', 'min', 'max', 'sum',
                'abs', 'clip', 'where', 'groupby', 'transform']
    for kw in keywords:
        if kw in formula:
            if kw == 'rolling':
                # 提取 rolling 后的聚合方法
                import re
                for m in re.finditer(r'rolling\(\d+\)\.(\w+)', formula):
                    if m.group(1) not in ops:
                        ops.append(f'rolling_{m.group(1)}')
            elif kw not in ops:
                ops.append(kw)
    return ','.join(ops[:5]) if ops else 'unknown'


def _extract_operators_list(formula: str) -> list:
    """同 _extract_operators，但返回列表。"""
    s = _extract_operators(formula)
    return s.split(',') if s != 'unknown' else []


def main():
    parser = argparse.ArgumentParser(description='Stage2 PASS → Stage3 因子库+Memory 桥接')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不写入')
    parser.add_argument('--factor', type=str, help='仅注入指定因子名')
    args = parser.parse_args()

    # 加载已有数据
    pool_names = load_existing_pool_names()
    attempt_names, template_names = load_existing_memory_names()
    pass_factors = load_pass_factors()

    if args.factor:
        pass_factors = [f for f in pass_factors if f.get('factor_name') == args.factor]
        if not pass_factors:
            print(f"未找到因子: {args.factor}")
            return

    print(f"Stage2 → Stage3 桥接")
    print(f"  因子库: {len(pool_names)} 已有 | Memory: {len(attempt_names)} attempts + {len(template_names)} templates")
    print(f"  PASS 候选: {len(pass_factors)} 个")

    # 读取 CSV fieldnames (用于写入)
    if POOL_CSV.exists():
        with open(POOL_CSV, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            csv_rows = list(reader)
    else:
        fieldnames = []
        csv_rows = []

    # 加载 Memory
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            mem_data = json.load(f)
    else:
        mem_data = {"attempts": [], "success_templates": [], "stats": {}}

    injected_pool = 0
    injected_attempt = 0
    skipped = 0

    for factor in pass_factors:
        name = factor.get('factor_name', '')

        # 检查是否已完全入库
        in_pool = name in pool_names
        in_attempt = name in attempt_names
        in_template = name in template_names

        if in_pool and in_attempt and in_template:
            skipped += 1
            continue

        print(f"\n  → {name}")
        print(f"    paradigm={factor.get('paradigm','?')} "
              f"icir={factor.get('icir','?')} pool={'✅' if in_pool else '❌'} "
              f"attempt={'✅' if in_attempt else '❌'} template={'✅' if in_template else '❌'}")

        if not in_pool and fieldnames:
            row = inject_to_pool(factor, pool_names, fieldnames)
            if row:
                csv_rows.append(row)
                pool_names.add(name)
                injected_pool += 1
                print(f"    + CSV")

        if not in_attempt:
            inject_to_memory(factor, attempt_names, template_names, mem_data)
            if name not in attempt_names:
                attempt_names.add(name)
                injected_attempt += 1
                print(f"    + attempt")

    # 写入
    if args.dry_run:
        print(f"\n[Dry-run] 将注入: {injected_pool} CSV + {injected_attempt} attempts "
              f"(v0.8: 模板池门槛 S5, Stage2 PASS 不再写 templates)")
        print(f"[Dry-run] 跳过: {skipped} 个已完全入库")
        return

    if injected_pool > 0:
        with open(POOL_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n✅ CSV 写入: +{injected_pool} → {len(csv_rows)} 行")

    if injected_attempt > 0:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(mem_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Memory 写入: +{injected_attempt} attempts → "
              f"{len(mem_data.get('attempts',[]))} attempts "
              f"(v0.8: 模板池准入门槛 S5, Stage2 PASS 不写 success_templates)")

    print(f"\n=== 桥接完成 ===")
    print(f"  注入: {injected_pool} 因子库 + {injected_attempt} attempts")
    print(f"  跳过: {skipped} 个已完全入库")


if __name__ == '__main__':
    main()
