"""
Stage 1 → Stage 2 因子提案桥接脚本

读取 stage1_optimization_proposals.json 中的「因子扩展」类提案，
用 LLM (DeepSeek) 自动生成含 pandas 公式的因子定义，
写入 stage1_factor_proposals.json 供 Stage 2 daily_factor_hypothesis.py 消费。

用法:
    python scripts/stage1_to_factor_proposals.py                        # 处理全部未完成的因子扩展提案
    python scripts/stage1_to_factor_proposals.py --proposal P-001      # 仅处理指定提案
    python scripts/stage1_to_factor_proposals.py --dry-run             # 预览模式，不实际写入
    python scripts/stage1_to_factor_proposals.py --max-per-proposal 3  # 每提案最多生成 N 个因子
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
OPTIMIZATION_PROPOSALS_FILE = PROJECT_ROOT / 'data' / 'stage1_optimization_proposals.json'
FACTOR_PROPOSALS_FILE = PROJECT_ROOT / 'data' / 'stage1_factor_proposals.json'

# ── LLM 接入 ──────────────────────────────────────────────

def _get_llm_client():
    """延迟导入，避免无 llm_client 时脚本直接崩溃。"""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'research' / 'factor_alchemy'))
        from llm_client import get_llm_client
        return get_llm_client()
    except Exception:
        return None


def _call_llm(prompt: str, temperature: float = 0.3) -> str | None:
    client = _get_llm_client()
    if client is None:
        return None
    try:
        resp = client.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], temperature=temperature)
        return resp
    except Exception as e:
        print(f"  ⚠️ LLM 调用失败: {e}")
        return None


SYSTEM_PROMPT = """你是 A 股量化因子专家。你的任务是把一个"因子研究方向描述"转化为 3-5 个具体的、可计算的 pandas 因子公式。

## ⚠️ 关键规则 — 违反即失败
1. **禁止使用 cs_rank()** — 绝对禁止! 用 .rank(pct=True) 代替
2. **禁止 Forge 风格函数** — ts_mean/ts_std/ts_delta/ts_pct/ts_zscore/ts_corr/neg/rank/div/sub/add/mul/scale/returns 这些全都不允许!
3. **禁止虚构列名** — rd_expense/rd_capitalized/rd_staff/revenue/north_money/south_money/goodwill/intan_assets/concept_label 等全部不存在!
4. **open/high/low/close 是列名不是函数** — 写成表达式如 close.rolling(20) 直接使用, 切勿写成 low()/high() 加括号的形式

## 可用数据字段 (只有这些!)
- OHLCV: open, high, low, close, volume, amount
- 资金流: buy_lg_vol, sell_lg_vol, buy_sm_vol, sell_sm_vol (buy=主动买入, sell=主动卖出, lg=大单, sm=小单)
- ⚠️ amount 仅当日频数据包含, 如无法使用可用 (high*close*volume)**0.5 代理

## 公式语法 (pandas 纯表达式, 单行)
- 窗口: .rolling(N).mean(), .rolling(N).std(), .rolling(N).min(), .rolling(N).max()
- 位移: .shift(N), .diff(N), .pct_change(N)
- 排名: .rank(pct=True) — 直接用, 不要用 cs_rank()
- 组合: .rolling(N).mean() → .rolling(N).std() → .rank(pct=True)
- 方向: 整个表达式前加负号 = 做多反向信号
- 去量纲: (X - X.rolling(N).mean()) / (X.rolling(N).std() + 1e-8)
- 防除零: 分母加 + 1e-8

## 正确示例
✅ (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
✅ (buy_lg_vol - sell_lg_vol).rolling(5).mean() / (volume.rolling(20).mean() + 1e-8)
✅ -(close.pct_change(5).rank(pct=True))
❌ cs_rank(close.pct_change(5))  ← 禁止!
❌ neg(rank(div(close, ts_mean(close, 20))))  ← Forge 风格禁止!
❌ rd_expense / revenue  ← 虚构列名禁止!

## 输出格式
必须严格输出 JSON 数组 (不要 markdown code block):
[{
  "factor_name": "snake_case_英文名",
  "label": "中文名",
  "formula_pandas": "pandas 纯表达式 (单行)",
  "economic_rationale": "经济逻辑 (1-2句)",
  "paradigm": "范式分类",
  "direction": "long 或 short"
}]

## 约束
- 每个因子必须有明确的金融经济学逻辑
- 公式只依赖上述可用字段 + pandas 操作
- 避免纯恒等变换 (如 close - close)
- 每个提案生成 3-5 个因子，不要重复
- 只输出裸 JSON 数组，不要有任何额外文字或 markdown 标记
"""


def build_llm_prompt(proposal: dict, existing_names: set) -> str:
    """根据提案构建 LLM 提示词。"""
    title = proposal.get('title', '')
    prop_change = proposal.get('proposed_change', '')
    trigger = proposal.get('trigger', '')
    refs = proposal.get('references', [])
    pid = proposal.get('proposal_id', '')

    ref_text = '\n'.join(f'- {r}' for r in refs[:3]) if refs else '(无)'

    return f"""## 提案: {pid} — {title}

### 研究方向
{prop_change}

### 触发原因/参考文献
{trigger}

### 参考文献
{ref_text}

### 已有因子名 (不可重复)
{', '.join(sorted(existing_names)[:30])}

请为这个提案生成 3-5 个具体因子公式。只输出 JSON 数组。"""


# ── 核心逻辑 ──────────────────────────────────────────────

def load_existing_factor_names() -> set:
    """加载已有因子名 (避免重名)。"""
    names = set()
    # 从 NOVEL_FACTOR_LIBRARY 读取
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
        from daily_factor_hypothesis import NOVEL_FACTOR_LIBRARY
        for f in NOVEL_FACTOR_LIBRARY:
            names.add(f.get('name', ''))
    except Exception:
        pass

    # 从已有 factor_proposals 读取
    if FACTOR_PROPOSALS_FILE.exists():
        try:
            data = json.loads(FACTOR_PROPOSALS_FILE.read_text(encoding='utf-8'))
            for p in data.get('proposals', []):
                names.add(p.get('factor_name', p.get('name', '')))
        except Exception:
            pass

    # 从因子池读取
    pool_file = PROJECT_ROOT / 'data' / 'library_orthogonality_state.json'
    if pool_file.exists():
        try:
            data = json.loads(pool_file.read_text(encoding='utf-8'))
            for f in data.get('factors', []):
                names.add(f.get('name', ''))
        except Exception:
            pass

    return names


def parse_llm_response(response: str, existing_names: set) -> list[dict]:
    """解析 LLM 返回的 JSON 并归一化。"""
    # 尝试提取 JSON 块
    import re
    # 清理 markdown code block
    response = re.sub(r'```(?:json)?\s*', '', response).strip()
    response = re.sub(r'```\s*$', '', response).strip()

    try:
        factors = json.loads(response)
    except json.JSONDecodeError:
        # 尝试提取数组
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            try:
                factors = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(factors, list):
        return []

    normalized = []
    for f in factors:
        name = f.get('factor_name', '')
        if not name:
            continue
        if name in existing_names:
            # 自动添加后缀避免重名
            name = f"{name}_v2"
        if name in existing_names:
            continue  # 仍重名则跳过

        normalized.append({
            'factor_name': name,
            'label': f.get('label', name),
            'formula_pandas': f.get('formula_pandas', f.get('formula', '')),
            'economic_rationale': f.get('economic_rationale', f.get('hypothesis', '')),
            'paradigm': f.get('paradigm', ''),
            'direction': f.get('direction', 'long'),
            'source_proposal': f.get('source_proposal', ''),
            'generated_at': datetime.now().isoformat(),
        })
        existing_names.add(name)

    return normalized


def process_proposals(proposal_ids: list[str] | None = None,
                      max_per_proposal: int = 5,
                      dry_run: bool = False) -> dict:
    """主处理逻辑。

    Returns:
        dict: {'processed': int, 'generated': int, 'errors': [str]}
    """
    if not OPTIMIZATION_PROPOSALS_FILE.exists():
        return {'processed': 0, 'generated': 0, 'errors': ['优化提案文件不存在']}

    data = json.loads(OPTIMIZATION_PROPOSALS_FILE.read_text(encoding='utf-8'))
    all_proposals = data.get('proposals', []) if isinstance(data, dict) else data

    # 筛选因子扩展类提案
    factor_ext = [p for p in all_proposals
                  if '因子' in p.get('category', '')]
    if proposal_ids:
        factor_ext = [p for p in factor_ext if p.get('proposal_id') in proposal_ids]

    if not factor_ext:
        print('没有因子扩展类提案需要处理.')
        return {'processed': 0, 'generated': 0, 'errors': []}

    existing_names = load_existing_factor_names()
    print(f'已有因子名: {len(existing_names)} 个')

    # 加载已有 factor proposals
    existing_proposals = []
    if FACTOR_PROPOSALS_FILE.exists():
        try:
            ed = json.loads(FACTOR_PROPOSALS_FILE.read_text(encoding='utf-8'))
            existing_proposals = ed.get('proposals', []) if isinstance(ed, dict) else ed
        except Exception:
            pass

    llm_available = _get_llm_client() is not None
    if not llm_available:
        print('⚠️ LLM 不可用, 将生成模板占位因子 (需后续人工填公式)')

    total_generated = 0
    errors = []

    for prop in factor_ext:
        pid = prop.get('proposal_id', '?')
        title = prop.get('title', pid)
        print(f'\n处理: {pid} — {title[:60]}')

        if llm_available:
            prompt = build_llm_prompt(prop, existing_names)
            response = _call_llm(prompt)
            if response:
                new_factors = parse_llm_response(response, existing_names)
                if new_factors:
                    for nf in new_factors:
                        nf['source_proposal'] = pid
                    existing_proposals.extend(new_factors)
                    total_generated += len(new_factors)
                    print(f'  ✅ 生成 {len(new_factors)} 个因子: {[f["factor_name"] for f in new_factors]}')
                else:
                    print(f'  ⚠️ LLM 响应解析失败 (前150字): {response[:150]}')
                    errors.append(f'{pid}: 解析失败')
            else:
                print(f'  ❌ LLM 调用失败')
                errors.append(f'{pid}: LLM调用失败')
        else:
            # 无 LLM 时生成占位模板
            placeholder_count = min(3, max_per_proposal)
            for i in range(placeholder_count):
                name = f"{pid.replace('-','_').lower()}_f{i+1}"
                if name in existing_names:
                    name = f"{name}_v2"
                existing_proposals.append({
                    'factor_name': name,
                    'label': f'{title[:30]}_因子{i+1}',
                    'formula_pandas': f'# TODO: 基于 "{title[:50]}" 的公式',
                    'economic_rationale': prop.get('proposed_change', '')[:100],
                    'paradigm': prop.get('trigger', '').split('范式')[-1].split(')')[0] if '范式' in prop.get('trigger', '') else '',
                    'direction': 'long',
                    'source_proposal': pid,
                    'generated_at': datetime.now().isoformat(),
                })
                existing_names.add(name)
            total_generated += placeholder_count
            print(f'  ⚠️ 生成 {placeholder_count} 个占位因子 (需人工填公式)')

    # 写入
    if not dry_run and total_generated > 0:
        output = {'proposals': existing_proposals, 'updated_at': datetime.now().isoformat()}
        FACTOR_PROPOSALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        FACTOR_PROPOSALS_FILE.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n✅ 已写入 {FACTOR_PROPOSALS_FILE}: {len(existing_proposals)} 个因子提案')
    elif dry_run:
        print(f'\n🔍 [预览] 将写入 {total_generated} 个因子 (未实际写入)')

    return {
        'processed': len(factor_ext),
        'generated': total_generated,
        'errors': errors,
    }


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Stage1→Stage2 因子提案桥接')
    parser.add_argument('--proposal', type=str, help='仅处理指定提案 ID')
    parser.add_argument('--dry-run', action='store_true', help='预览模式')
    parser.add_argument('--max-per-proposal', type=int, default=5,
                        help='每提案最多生成因子数 (默认 5)')
    args = parser.parse_args()

    proposal_ids = [args.proposal] if args.proposal else None

    result = process_proposals(
        proposal_ids=proposal_ids,
        max_per_proposal=args.max_per_proposal,
        dry_run=args.dry_run,
    )

    print(f'\n=== 汇总 ===')
    print(f'处理提案: {result["processed"]}')
    print(f'生成因子: {result["generated"]}')
    if result['errors']:
        print(f'错误: {len(result["errors"])}')
        for e in result['errors']:
            print(f'  - {e}')


if __name__ == '__main__':
    main()
