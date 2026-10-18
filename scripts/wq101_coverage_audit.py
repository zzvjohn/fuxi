# -*- coding: utf-8 -*-
"""P-20260908-002: WorldQuant 101 算子覆盖审计 + 缺口模板注入

审计 NOVEL_FACTOR_LIBRARY (scripts/daily_factor_hypothesis.py 内嵌 201 条) 对
WQ101 十一类经典算子的 pandas 等价物覆盖率, 输出缺口清单, 并为缺口类生成
内置 pandas 风格补充模板 (避开 hard forbidden 与已证伪方向)。

输出: data/wq101_coverage_audit.json
用法: python scripts/wq101_coverage_audit.py [--inject]
       --inject: 将缺口模板追加进 daily_factor_hypothesis.py 的 NOVEL_FACTOR_LIBRARY
"""
import ast
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / 'scripts' / 'daily_factor_hypothesis.py'
OUT = PROJECT_ROOT / 'data' / 'wq101_coverage_audit.json'

# ── WQ101 十一类算子 → pandas 等价物正则 ──
OPERATOR_CLASSES = {
    'ts_rank_时序排名': [
        re.compile(r'\.rolling\(\d+\)\.rank\('),
    ],
    'ts_stat_时序统计(mean/std/sum/min/max)': [
        re.compile(r'\.rolling\(\d+\)\.(mean|std|sum|min|max|var|skew|kurt)\('),
    ],
    'rank_截面排名': [
        re.compile(r'\.rank\(pct=True\)'),          # 裸 rank 预处理已强制 axis=1
        re.compile(r'groupby\([^)]*\)[^;]*\.rank\('),
    ],
    'delta_时序差分': [
        re.compile(r'\.diff\(\d*\)'),
        re.compile(r'-\s*[A-Za-z_][\w_]*\.shift\(\d*\)'),
    ],
    'ts_corr_时序相关': [
        re.compile(r'\.rolling\(\d+\)\.corr\('),
    ],
    'ts_decay_衰减加权': [
        re.compile(r'np\.exp\('),
        re.compile(r'decay|weighted|ewm\('),
    ],
    'scale_缩放归一': [
        re.compile(r'\.abs\(\)\.rolling\(\d+\)\.(sum|mean)\('),
        re.compile(r'\.rolling\(\d+\)\.std\(\)\s*\)?\s*$'),
    ],
    'indneutralize_行业中性化': [
        re.compile(r'groupby\([^)]*\)\.transform\('),
        re.compile(r'\.sub\([^)]*groupby'),
    ],
    'advance_decay_涨跌家数': [
        re.compile(r'\(pct_chg\s*[><]=?\s*0\)'),
        re.compile(r'\.clip\(lower=0\)'),
    ],
    'signedpower_符号幂': [
        re.compile(r'np\.sign\([^)]*\)\s*\*\s*np\.abs\([^)]*\)\s*\*\*'),
    ],
    'cross_corr_截面相关': [
        re.compile(r'groupby\([^)]*\)\s*\.corr'),
    ],
}

# ── 缺口类内置补充模板 (pandas .rolling/.shift/.rank 风格, 已避开硬禁方向) ──
# 注意: 全部为评估宽表环境可 eval 的形式 (无 industry/trade_date 列依赖, 无裸 np)
GAP_TEMPLATES = {
    'ts_rank_时序排名': [
        {
            'name': 'ts_rank_momentum_60_20',
            'label': '时序排名动量(60日内20日收益排名)',
            'category': 'ts_rank',
            'hypothesis': '个股自身时序相对强弱的位置比绝对动量更稳健',
            'logic': 'WQ101 ts_rank 族: 自身时序排名标准化动量, 滤除个股波动水平差异',
            'formula': 'close_p.pct_change(20).rolling(60).rank(pct=True)',
            'direction': 'long',
        },
    ],
    'ts_corr_时序相关': [
        {
            'name': 'ts_corr_vol_ret_60',
            'label': '量价时序相关(60日成交量与收益率)',
            'category': 'ts_corr',
            'hypothesis': '放量上涨一致性的个股有趋势质量, 相关性转负为派发信号',
            'logic': 'WQ101 ts_corr 族: volume 与收益的时序相关捕捉量价配合度 (fillna(0) 防 rolling corr NaN 铁律)',
            'formula': 'volume_p.rolling(60).corr(close_p.pct_change().fillna(0))',
            'direction': 'long',
        },
    ],
    'ts_decay_衰减加权': [
        {
            'name': 'ts_decay_ewm_halflife10',
            'label': '指数衰减加权收益(ewm 半衰期10日)',
            'category': 'ts_decay',
            'hypothesis': '近端收益信息权重大于远端, 衰减加权比等权动量响应更快',
            'logic': 'WQ101 decay_linear 族: ewm 指数衰减权重替代等权 rolling mean (纯 pandas, 无 np 依赖)',
            'formula': 'close_p.pct_change().ewm(halflife=10).mean()',
            'direction': 'long',
        },
    ],
    'signedpower_符号幂': [
        {
            'name': 'signedpower_ret_20_15',
            'label': '符号幂收益(20日收益^1.5保符号)',
            'category': 'signedpower',
            'hypothesis': '保符号幂放大大幅波动股的信号强度, 小波动股被压缩',
            'logic': 'WQ101 signedpower 族: sign(x)*|x|^a 对极端收益敏感化 (x/|x| 实现符号, 无 np 依赖)',
            'formula': '((close_p.pct_change(20) / (close_p.pct_change(20).abs() + 1e-8)) * close_p.pct_change(20).abs() ** 1.5)',
            'direction': 'long',
        },
    ],
    'indneutralize_行业中性化': [
        {
            'name': 'mkt_neutralized_momentum_20',
            'label': '市场中性化动量(20日超额市场动量, 行业中性待 P-001 升级)',
            'category': 'indneutralize',
            'hypothesis': '剔除市场 Beta 后的动量是个股特有信息, 拥挤度更低',
            'logic': 'WQ101 indneutralize 族降级版: 宽表 .mean(axis=1) 减全市场截面均值; 行业级中性化待 P-20260908-001 sw 行业映射接入后升级',
            'formula': '(close_p.pct_change(20) - close_p.pct_change(20).mean(axis=1))',
            'direction': 'long',
        },
    ],
    'advance_decay_涨跌家数': [
        {
            'name': 'advance_days_ratio_5d',
            'label': '个股上涨天数占比(5日)',
            'category': 'advance_decay',
            'hypothesis': '近期上涨天数占比高的个股趋势确认度高',
            'logic': 'WQ101 advance 族个股级变体: 涨跌计数 (市场级宽度因子待分域通道接入)',
            'formula': '(close_p.pct_change() > 0).astype(float).rolling(5).mean()',
            'direction': 'long',
        },
    ],
    # cross_corr_截面相关: 无评估环境安全的宽表形式 (需市场级环境), 留待 P-001 分域通道, 审计报告如实记录
}


def extract_library():
    """AST 提取 NOVEL_FACTOR_LIBRARY 的 (name, formula) 列表"""
    src = SRC.read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'NOVEL_FACTOR_LIBRARY':
                    items = []
                    for e in node.value.elts:
                        if isinstance(e, ast.Dict):
                            d = {}
                            for k, v in zip(e.keys, e.values):
                                if isinstance(k, ast.Constant):
                                    d[k.value] = (v.value if isinstance(v, ast.Constant)
                                                  else ast.unparse(v))
                            items.append(d)
                    return items
    return []


def audit():
    lib = extract_library()
    formulas = [x.get('formula', '') for x in lib]
    report = {
        'generated_at': None,
        'library_size': len(lib),
        'operator_classes': {},
        'gaps': [],
    }
    for cls, pats in OPERATOR_CLASSES.items():
        hits = []
        for f in formulas:
            if any(p.search(f) for p in pats):
                hits.append(f)
        n = len(hits)
        report['operator_classes'][cls] = {
            'hit_templates': n,
            'coverage': round(n / max(len(formulas), 1), 4),
            'examples': hits[:2],
        }
        if n == 0:
            report['gaps'].append(cls)
    return report, lib


def do_inject(lib, gaps):
    src = SRC.read_text(encoding='utf-8')
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'NOVEL_FACTOR_LIBRARY':
                    target = node
    last = target.value.elts[-1]
    lines = src.splitlines(keepends=True)
    insert_idx = last.end_lineno  # 0-based 下一行

    existing_names = {x.get('name') for x in lib}
    max_id = 0
    for x in lib:
        m = re.match(r'novel_(\d+)', str(x.get('id', '')))
        if m:
            max_id = max(max_id, int(m.group(1)))

    blocks = []
    for cls in gaps:
        for tmpl in GAP_TEMPLATES.get(cls, []):
            if tmpl['name'] in existing_names:
                continue
            max_id += 1
            entry = {
                'id': f'novel_{max_id:03d}',
                'name': tmpl['name'],
                'label': tmpl['label'],
                'category': tmpl['category'],
                'hypothesis': tmpl['hypothesis'],
                'logic': tmpl['logic'],
                'formula': tmpl['formula'],
                'direction': tmpl['direction'],
            }
            blocks.append(entry)
    if not blocks:
        return 0

    # 序列化为与现文件一致的 dict 字面量
    import pprint
    rendered = []
    for b in blocks:
        s = '    {\n'
        for k, v in b.items():
            s += f'        {k!r}: {v!r},\n'
        s += '    },\n'
        rendered.append(s)
    # 找到末尾 ] 前的插入点 (在 insert_idx 行前)
    chunk = ''.join(rendered)
    lines.insert(insert_idx, chunk)
    SRC.write_text(''.join(lines), encoding='utf-8')
    return len(blocks)


def main():
    report, lib = audit()
    report['generated_at'] = __import__('datetime').datetime.now().isoformat()
    gaps = report['gaps']
    print(f"[WQ101审计] 库模板 {len(lib)} 条")
    for cls, info in report['operator_classes'].items():
        flag = '  ← 缺口' if info['hit_templates'] == 0 else ''
        print(f"  {cls}: {info['hit_templates']} 条覆盖 ({info['coverage']:.1%}){flag}")
    print(f"[WQ101审计] 缺口类 {len(gaps)}: {gaps}")

    if '--inject' in sys.argv:
        n = do_inject(lib, gaps)
        print(f"[WQ101注入] 已追加 {n} 条缺口模板进 NOVEL_FACTOR_LIBRARY")
        report['injected'] = n
        # 重新审计验证
        lib2 = extract_library()
        report['library_size_after'] = len(lib2)
        print(f"[WQ101注入] 库规模 {len(lib)} → {len(lib2)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(f"[WQ101审计] 落盘 {OUT}")


if __name__ == '__main__':
    main()
