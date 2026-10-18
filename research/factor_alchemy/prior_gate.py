# -*- coding: utf-8 -*-
"""P-20260914-003: 因子经济学先验门槛 (影子 — 只打标 + 落盘, 默认零裁决)

方法论 (Two Sigma 特征引擎访谈 × AQuA arXiv 相互论证, cross_verified=true):
  Two Sigma: 「所有因子必须经过经济学逻辑检验, 坚决排除缺乏商业逻辑的低先验指标」
  AQuA: agent 记忆记录候选因子被提出的原因与推翻证据
  ⇒ 生成端候选按「机制说明质量」打 prior 标签, 影子观测
    prior 等级与 S1 通过 / S5 拒绝的相关性; 若『无逻辑先验的纯统计因子
    S5 通过率显著更低』被实证, 再考虑转 enforce (软门槛, 降优先级不硬拒)。

分级 (启发式, 首版, 零 LLM 配额):
  prior_strong : 机制关键词 + 可证伪方向 token 双全
  prior_medium : 二者有其一
  prior_weak   : 双缺 或 机制文本过短 (<15 字符)
  机制关键词: 资金/筹码/情绪/估值/供需/动量/反转/流动性/波动/拥挤/盈利/成长/
             行为/风险/惯性/博弈/均值回归/溢价/贴现/机构/散户 ...
  方向 token: 为正/为负/上涨/下跌/正相关/负相关/更高/更低/跑赢/跑输 ...

模式: env FUXI_PRIOR_GATE ∈ off | shadow (默认) | on
  off    : 完全跳过
  shadow : 评估 + 落盘 data/prior_gate_report.json + stdout 读数, 不改任何裁决
  on     : 同 shadow + prior_weak 候选在结果对象上标记 (供未来软门槛接线, 当前仍不裁决)
可选: env FUXI_PRIOR_LLM=1 时启用 LLM 机制补写 (额外配额, 默认关闭 — 首版启发式足够)。

回滚: env FUXI_PRIOR_GATE=off 即完全禁用; 删除本文件 + ralph_loop 钩子即彻底移除。
"""
import os
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
if not DATA_DIR.exists():
    DATA_DIR = Path('e:/quant/data')  # 兜底: 从 ralph_loop 导入时 data_dir 由调用方传入
REPORT_FILE = DATA_DIR / 'prior_gate_report.json'
MAX_BATCHES = 200   # 报告只保留最近 N 批, 防膨胀

MECH_KW = re.compile(
    r'资金|筹码|情绪|估值|供需|动量|反转|流动性|波动|拥挤|盈利|成长|行为|风险|'
    r'惯性|博弈|均值回归|溢价|贴现|折价|机构|散户|主力|承接|换手|套利|利差|'
    r'liquidity|momentum|reversal|crowding|valuation|fundamental|flow|behavior',
    re.IGNORECASE)
DIR_KW = re.compile(
    r'为正|为负|上涨|下跌|正相关|负相关|更高|更低|跑赢|跑输|高于|低于|'
    r'positive|negative|outperform|underperform|higher|lower',
    re.IGNORECASE)


def get_mode():
    return (os.environ.get('FUXI_PRIOR_GATE') or 'shadow').strip().lower()


def evaluate_candidate(cand: dict) -> dict:
    """对单条候选做机制质量分级 (启发式, 零 LLM 调用)"""
    text = ' '.join(str(cand.get(k) or '') for k in
                    ('hypothesis', 'logic', 'mechanism', 'label', 'rationale')).strip()
    name = cand.get('factor_name') or cand.get('name') or ''
    has_mech = bool(MECH_KW.search(text))
    has_dir = bool(DIR_KW.search(text))
    if len(text) < 15:
        grade, reason = 'prior_weak', 'no_mechanism_text'
    elif has_mech and has_dir:
        grade, reason = 'prior_strong', 'mechanism+falsifiable_direction'
    elif has_mech or has_dir:
        grade, reason = 'prior_medium', 'mechanism_xor_direction'
    else:
        grade, reason = 'prior_weak', 'no_mechanism_nor_direction'
    return {
        'name': name,
        'grade': grade,
        'reason': reason,
        'mechanism_text': text[:160],
        'predicted_direction': cand.get('direction', ''),
        'text_len': len(text),
    }


def run_prior_gate(candidates: list, evaluate_result: dict, data_dir=None,
                   generator: str = '') -> dict:
    """ralph_loop 钩子入口: 对本轮全部候选打 prior 标签并落盘影子报告。

    on 模式下在 ValidationResult 对象上附加 prior_* 属性 (供未来软门槛接线,
    当前任何模式都不改 s*_passed / eligible_for_jq 等裁决字段)。
    """
    mode = get_mode()
    cands = candidates or []
    if mode == 'off' or not cands:
        return {'mode': mode, 'n': 0, 'skipped': True}

    results = {getattr(r, 'factor_name', ''): r
               for r in evaluate_result.get('results', [])}
    evals = [evaluate_candidate(c) for c in cands]

    # on 模式: 附加属性 (dataclass 无 slots 时生效; 失败静默忽略)
    if mode == 'on':
        for ev in evals:
            r = results.get(ev['name'])
            if r is not None:
                try:
                    setattr(r, 'prior_grade', ev['grade'])
                    setattr(r, 'prior_reason', ev['reason'])
                except Exception:  # noqa: BLE001
                    pass

    # 影子读数: prior 等级 × S1 通过 交叉表 (观测『无先验是否更难过关』)
    cross = {}
    for ev in evals:
        r = results.get(ev['name'])
        s1 = bool(getattr(r, 's1_passed', False)) if r else None
        key = f"{ev['grade']}|s1={'pass' if s1 else ('fail' if s1 is False else 'na')}"
        cross[key] = cross.get(key, 0) + 1

    batch = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'generator': generator,
        'n': len(evals),
        'summary': {g: sum(1 for e in evals if e['grade'] == g)
                    for g in ('prior_strong', 'prior_medium', 'prior_weak')},
        'cross_s1': cross,
        'evals': evals,
    }

    out_dir = Path(data_dir) if data_dir else DATA_DIR
    report = out_dir / 'prior_gate_report.json'
    try:
        hist = json.loads(report.read_text(encoding='utf-8')) if report.exists() else []
        hist.append(batch)
        report.write_text(json.dumps(hist[-MAX_BATCHES:], ensure_ascii=False, indent=2),
                          encoding='utf-8')
    except Exception as e:  # noqa: BLE001 — 落盘失败不阻塞主流程
        print(f"  [PriorGate] 报告落盘失败 (非阻塞): {e}")

    s = batch['summary']
    print(f"  [PriorGate:{mode}] {len(evals)} 候选 → strong {s['prior_strong']} / "
          f"medium {s['prior_medium']} / weak {s['prior_weak']} "
          f"(gen={generator or '?'})")
    return batch


if __name__ == '__main__':
    # 自检: 三条合成候选覆盖三个等级
    demo = [
        {'factor_name': 'a_strong', 'direction': 'long',
         'hypothesis': '机构资金持续净流入锁定筹码, 未来5日超额收益为正',
         'logic': '资金流持续性与动量'},
        {'factor_name': 'b_medium', 'direction': 'short',
         'hypothesis': '拥挤度高的因子族收益衰减', 'logic': ''},
        {'factor_name': 'c_weak', 'direction': '', 'hypothesis': '', 'logic': ''},
    ]
    print('模式:', get_mode())
    for d in demo:
        ev = evaluate_candidate(d)
        print(f"  {ev['name']:10s} → {ev['grade']:13s} ({ev['reason']})")
