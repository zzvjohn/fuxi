# -*- coding: utf-8 -*-
"""P-20260914-002: 机械 vs 判断因子衰减异质性分类标签 (纯影子 — 只打标签, 零裁决)

方法论 (arXiv:2512.11913《Not All Factors Crowd Equally》):
  博弈论模型推导 alpha 衰减为 hyperbolic α(t)=K/(1+λt), 并实证
  「机械因子(信号无歧义易复制, 如动量)符合 hyperbolic 衰减 R²=0.65,
    判断因子(信号多义, 如价值)不符合」。
  ⇒ 库内因子按公式结构自动打 mechanism_class 标签, 供 DecayMonitor
  未来按类分层设定衰减参数 (当前不改任何裁决, 只观测)。

分类规则 (启发式, 首版):
  judgment 优先命中:
    1. expression/名称含主观字段关键词 (rating/sentiment/news/analyst/评级/情绪/研报/舆情)
    2. expression 引用基本面/估值/资金流字段 (roe|pb|pe|eps|n_income|total_mv|kd_*|mf_*|buy_*/sell_*)
       —— 价值/质量/资金类 = 信号多义, 对应论文的 judgment 类
    3. 算子数 >= 5 (.rolling/.shift/.rank/.ewm/.expanding/.apply 计数) —— 复杂链式 = 多义
  否则 → mechanical (少算子纯 OHLCV, 信号无歧义易复制)

输出: data/mechanism_class_tags.json + stdout 读数
      [MechClass] N 因子 → mechanical X (Y%) / judgment Z
用法: python scripts/factor_mechanism_class.py [--state data/library_orthogonality_state.json]
回滚: 纯标签脚本, 删除即完全移除; 不写回库状态。
"""
import sys
import json
import re
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUT_FILE = DATA_DIR / 'mechanism_class_tags.json'

SUBJECTIVE_KW = re.compile(
    r'rating|sentiment|news|analyst|评级|情绪|研报|舆情|微博|搜索', re.IGNORECASE)
FUNDAMENTAL_FIELD = re.compile(
    r'\b(roe|pb|pe|ps|eps|n_income|total_mv|money_cap|total_assets|debt|'
    r'cashflow|revenue|dividend|repurchase|holder|turnover_rate|free_share)\b|'
    r'\b(kd_|mf_|buy_|sell_|net_|elg_|lg_|sm_|mid_)\w+', re.IGNORECASE)
OPS = ['.rolling(', '.shift(', '.rank(', '.ewm(', '.expanding(', '.apply(']


def classify(entry: dict):
    """返回 (mechanical_class, op_count, reason)"""
    expr = entry.get('expression') or entry.get('formula') or ''
    name = entry.get('name', '')
    op_count = sum(expr.count(op) for op in OPS)
    if SUBJECTIVE_KW.search(expr) or SUBJECTIVE_KW.search(name):
        return 'judgment', op_count, 'subjective_field'
    if FUNDAMENTAL_FIELD.search(expr):
        return 'judgment', op_count, 'fundamental_or_flow_field'
    if op_count >= 5:
        return 'judgment', op_count, 'complex_chain(op>=5)'
    return 'mechanical', op_count, 'simple_ohlcv(op<5,price/volume_only)'


def main():
    ap = argparse.ArgumentParser(description='P-20260914-002 机械/判断因子分类标签 (影子)')
    ap.add_argument('--state', type=str,
                    default=str(DATA_DIR / 'library_orthogonality_state.json'))
    args = ap.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"[MechClass] 库状态不存在: {state_path} → 跳过")
        return 1
    state = json.loads(state_path.read_text(encoding='utf-8'))
    factors = [f for f in state.get('factors', []) if f.get('name')]
    if not factors:
        print("[MechClass] 库为空 → 跳过")
        return 0

    tags, summary = {}, {'mechanical': 0, 'judgment': 0}
    reasons = {}
    for f in factors:
        cls, op_count, reason = classify(f)
        tags[f['name']] = {
            'mechanism_class': cls,
            'op_count': op_count,
            'reason': reason,
            'paradigm': f.get('paradigm', ''),
            'status': f.get('status', ''),
        }
        summary[cls] += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    n = len(factors)
    out = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'proposal': 'P-20260914-002',
        'mode': 'shadow',
        'reference': 'arXiv:2512.11913 机械因子 hyperbolic 衰减 R²=0.65 vs 判断因子不符合',
        'total': n,
        'summary': summary,
        'reason_breakdown': reasons,
        'tags': tags,
        'note': 'DecayMonitor 分层衰减参数为未来工作; 当前仅观测两类因子 S5 拒绝率差异',
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    pct = round(summary['mechanical'] / n * 100, 1)
    print(f"[MechClass] {n} 因子 → mechanical {summary['mechanical']} ({pct}%) / "
          f"judgment {summary['judgment']} (输出: {OUT_FILE.name})")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r}: {c}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
