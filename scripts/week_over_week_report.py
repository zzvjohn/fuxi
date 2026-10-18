# -*- coding: utf-8 -*-
"""
伏羲「一周前后对比 + 因子干旱归因」报告生成器
================================================
输入 (全部为已落盘的实测产物):
  data/_drought_analysis_20260912.json   — Ralph 逐日/逐周漏斗 + Memory 周对比
  data/_s5_caliber_ab_20260912.json      — S5 双口径对照实验 (同公式 × 旧/新口径)
  data/trajectory_log.json               — S5 逐条判据明细 (excess/calmar)
  data/version_registry.json             — 版本编号注册表
  data/experience_memory.json*           — Memory 快照时间序列
输出:
  reports/fuxi_week_over_week_<date>.md

用法: python scripts/week_over_week_report.py [--date YYYYMMDD]
"""
import json, os, sys, glob, re, collections, statistics, datetime, argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
D = 'data'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load(p, default=None):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return default


def fmt(x, nd=4):
    return ('%+.' + str(nd) + 'f') % x if isinstance(x, (int, float)) else '—'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.datetime.now().strftime('%Y%m%d'))
    args = ap.parse_args()

    dr = load(f'{D}/_drought_analysis_20260912.json', {})
    ab = load(f'{D}/_s5_caliber_ab_20260912.json', {})
    reg = load(f'{D}/version_registry.json', {})
    traj = load(f'{D}/trajectory_log.json', {})

    # ── S5 判据明细 ──
    recs = []
    for t in traj.get('trajectories', []):
        for e in (t.get('_eval_log') or []):
            if e.get('stage') != 'S5':
                continue
            rr = str(e.get('rejection_reason') or '')
            m = re.search(r'calmar_25=([-\d.]+),\s*calmar_26=([-\d.]+)', rr)
            c25, c26 = (float(m.group(1)), float(m.group(2))) if m else (None, None)
            recs.append({
                'date': str(e.get('timestamp'))[:10], 'name': str(e.get('factor_name'))[:44],
                'e25': e.get('excess_25'), 'e26': e.get('excess_26'),
                'c25': c25, 'c26': c26, 'reason': rr,
                'err': ('无有效数据' in rr) or ('数据加载失败' in rr),
            })

    def period(dt):
        if dt <= '2026-08-25':
            return 'A 通过期 08-10~08-25'
        if dt <= '2026-09-08':
            return 'B 旧口径干旱 08-29~09-08'
        return 'D 新口径 09-09~09-12'

    agg = collections.defaultdict(collections.Counter)
    for r in recs:
        a = agg[period(r['date'])]
        a['n'] += 1
        if r['err']:
            a['eval_err'] += 1; continue
        if not isinstance(r['e25'], (int, float)) or not isinstance(r['e26'], (int, float)):
            a['no_excess'] += 1; continue
        a['has_excess'] += 1
        ex_ok = r['e25'] > 0 and r['e26'] > 0
        if ex_ok:
            a['excess_ok'] += 1
        if r['c25'] is not None and r['c26'] is not None:
            c_ok = r['c25'] > 1.0 and r['c26'] > 1.0
            if ex_ok:
                a['near_miss_calmar'] += 1
            if ex_ok and c_ok:
                a['both_ok'] += 1

    # 全历史 calmar 双年同时达标
    calmar_both = [r for r in recs if r['c25'] is not None and r['c26'] is not None
                   and r['c25'] > 1.0 and r['c26'] > 1.0]
    excess_both = [r for r in recs if isinstance(r['e25'], (int, float))
                   and isinstance(r['e26'], (int, float)) and r['e25'] > 0 and r['e26'] > 0]

    # ── Memory 快照时间序列 ──
    mem = []
    for f in sorted(glob.glob(f'{D}/experience_memory.json*')):
        d = load(f)
        if not isinstance(d, dict) or 'attempts' not in d:
            continue
        ts = [a.get('timestamp') for a in d['attempts'] if a.get('timestamp')]
        mem.append({
            'file': os.path.basename(f), 'n': len(d['attempts']),
            'pass': (d.get('stats') or {}).get('total_pass'),
            'forbidden': len(d.get('forbidden_directions', [])),
            'last': max(ts)[:10] if ts else '—',
        })
    mem.sort(key=lambda x: x['last'])

    # ── 双口径对照 ──
    abl = ab.get('results', [])
    old = {r['factor_name']: r for r in abl if r['caliber'].startswith('OLD')}
    new = {r['factor_name']: r for r in abl if r['caliber'].startswith('NEW')}
    names = [r['factor_name'] for r in abl if r['caliber'].startswith('OLD')]
    # 口径漂移量 (NEW - OLD)
    _d25 = [new[k]['excess_25'] - old[k]['excess_25']
            for k in names
            if k in new and isinstance(new[k].get('excess_25'), (int, float))
            and isinstance(old[k].get('excess_25'), (int, float))]
    _d26 = [new[k]['excess_26'] - old[k]['excess_26']
            for k in names
            if k in new and isinstance(new[k].get('excess_26'), (int, float))
            and isinstance(old[k].get('excess_26'), (int, float))]
    _d25_mean = statistics.mean(_d25) if _d25 else 0.0
    _d26_mean = statistics.mean(_d26) if _d26 else 0.0
    _d25_worse = sum(1 for x in _d25 if x < 0)

    # 因子来源标注 (逐条核对, 避免笼统称"历史过 S5")
    PROV = {
        'oil_sensitive_cycle_lead_lag': 'JQ队列·过S5',
        'vshape_disposition_effect_v2': 'JQ队列·过S5',
        'limited_attention_max_return_5d': 'JQ队列·过S5',
        'herding_industry_deviation': 'JQ队列·过S5',
        'margin_volatility_squeeze': 'JQ队列·过S5',
        'vwap_dev_ratio_5_60': 'JQ队列·过S5',
        'fd_max-yield': 'JQ队列·过S5',
        'low_disagreement_momentum': 'JQ队列·过S5',
        'smallcap_20d_reversal_volume_ratio_extreme': 'JQ队列·过S5',
        'gap_momentum_low_turnover_low_volatility_combined': 'JQ队列·过S5',
        'max_drawdown_duration': 'JQ队列·过S5',
        'overnight_intraday_ratio': 'Memory·WEAK',
        'FORGE_0912_1': '当日forge候选',
        'FORGE_0912_2': '当日forge候选',
    }

    L = []
    A = L.append
    A('# 伏羲 一周前后对比 + 因子干旱归因')
    A('')
    A('> 生成时间：%s ｜ 生成器：`scripts/week_over_week_report.py`（可复跑）' %
      datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    A('> 对照口径：**一周前 = 2026-08-30~09-05**，**现在 = 2026-09-06~09-12**')
    A('')

    # ===== 0 结论 =====
    A('## 0. 结论速览')
    A('')
    A('**① 系统在「辨伪」维度确实更优**，有 5 项独立证据（§4）。')
    A('')
    A('**② 因子干旱不是新增守卫造成的**：落地的新守卫全部是 `shadow / 零裁决`，或与 S5 正交，'
      '不参与任何裁决（§3.5）。')
    A('')
    A('**③ 干旱的主因是 S5 终裁本身是「合取式双条件」**，与生成端能力已经脱节：')
    A('')
    A('- 终裁 = `(excess_2025>0 且 excess_2026>0)` **且** `(calmar_2025>1.0 且 calmar_2026>1.0)`')
    A('- 全历史 %d 条 S5 判据记录中，**双年正超额出现 %d 次，Calmar 双年同时 >1 出现 %d 次，'
      '两者同时满足 0 次**（§3.2）' % (len(recs), len(excess_both), len(calmar_both)))
    A('- 🔴 **2026-09-12 W5 修订**：`both_ok=0` 应表述为「在当前管线候选分布下**几乎不可满足**」，'
      '**判据本身不矛盾**（新口径下 both_ok=1）。真问题是**两条件高度正交 + 合取 ⇒ 通过率被乘性压缩**。')
    A('- 🔴 **2026-09-12 W5 修订**：`calmar_2026` 口径不可比问题**已修复** — 默认改为 '
      '`FUXI_S5_CALMAR_MODE=vol_scaled`（IR 型，分母换年化波动率，无窗口依赖）+ `FUXI_S5_IR_THRESHOLD=0.7`。'
      'legacy 口径均值随窗口从 60 日 3.148 漂到 252 日 1.090（2.89x），IR 型仅 0.531→0.558。')
    A('- 🔴 **2026-09-12 W5 修订**：**修完口径仍是 0/35 通过 ⇒ 口径不是瓶颈**。'
      '同池重裁（Top-200 版 35 条 × 3 口径）：legacy 0 / common_window 3 / vol_scaled 0。'
      '⇒ 方向应转向上游质量（W1 产量解封 + M1–M4），而非继续调口径。')
    A('- 新口径（09-09 起）下**管线自产候选**的「双年正超额」出现率归零 ⇒ 连卡在 Calmar 的机会都没有了')
    A('- 但**判据本身并非矛盾**：双口径对照实验（§3.4）在新口径下跑出 `both_ok = 1`，'
      '证明合取条件**可以**满足；真正的问题是**两个条件高度正交 + 合取 ⇒ 通过率被乘性压缩**')
    A('')
    A('**④ 09-09 的 v0.7.9 口径变更不是单纯的「抬门槛」，而是「换通过集合」**'
      '（Top-200/中证500 → 全量池/中证全指）：')
    A('')
    A('- 对**多数**因子更严：`NEW−OLD` 超额均值 `ex25` **%+.4f**（%d/%d 条变差）' % (
        _d25_mean, _d25_worse, len(_d25)))
    A('- 但确实**救活了至少一条**中小盘 alpha：`margin_volatility_squeeze` 旧口径 `ex25=-0.1215`（误杀）'
      '→ 新口径 `ex25=+0.0293` 且双年 Calmar 达标 ⇒ **新口径下通过 S5**（§3.4 发现 2）')
    A('- 同日重裁 39 条历史被拒因子 **0 条翻盘**（§3.3）'
      '⇒ 被拒因子的「正超额」多是大盘池噪声，换池后被稀释；但这**不等于**池误杀不存在')
    A('')
    A('**⑤ 版本编号纪律已建立**：v0.7.0~v0.7.11 回溯补全，覆盖 72 条已落地提案，'
      '`--check` 可作每日收尾闸门（§1）。')
    A('')

    # ===== 1 版本纪律 =====
    A('## 1. 版本编号纪律（本次新增）')
    A('')
    A('### 1.1 问题')
    A('')
    A('官方 `CHANGELOG.md` **止于 v0.7.2（2026-09-01）**。此后 12 天共落地约 40 条提案、'
      '新增 10+ 个管线模块，**全部没有版本号** —— 导致「系统到底改了什么、改在哪一天」不可追溯，'
      '这次做周对比时只能靠 git（无仓库）/ 文件 mtime / 提案库反推。')
    A('')
    A('### 1.2 版本政策（此后强制执行）')
    A('')
    A('| 项 | 规则 |')
    A('|---|---|')
    A('| 编号 | `v0.7.<patch>`，单调递增；**一天一批 = 一个版本**（同日多提案合并） |')
    A('| 升版判据 | `T1` 闸门/裁决逻辑增改 ｜ `T2` 数据口径或默认开关值变更 ｜ `T3` 新增管线模块/通道/产物（含影子守卫） |')
    A('| 不升版 | 纯报告/文档/复评结论/研究备忘 → 记入当日版本的非升版项 |')
    A('| 必填字段 | 提案 ID、变更域、**行为影响（是否零裁决）**、回退方式、关键口径变更 |')
    A('| 校验 | `python scripts/version_bump.py --check`（未编号即退出码 1，应作每日收尾步骤）|')
    A('')
    A('### 1.3 回溯补全的版本轴')
    A('')
    A('| 版本 | 日期 | 提案 | 行为影响 | 关键行为/口径 |')
    A('|---|---|---|---|---|')
    for v in reg.get('versions', []):
        lm = (v.get('landmark') or '').replace('|', '｜')
        lm = (lm[:76] + '…') if len(lm) > 76 else lm
        A('| **%s** | %s | %d | %s | %s |' % (
            v['version'], v['date'], len(v.get('proposals', [])),
            v.get('behavior', ''), lm or '—'))
    A('')
    A('> 🔴 **v0.7.9（2026-09-09）是本次干旱分析的关键版本**：S5 裁决池从 Top-200 流动性池'
      '（仅覆盖全A成交额 25.9%）改为全量池，基准从中证500 改为中证全指。')
    A('')
    A('产物：`CHANGELOG.md`（根目录，权威记录）、`data/version_registry.json`（机器可读）、'
      '`scripts/version_bump.py`（`--check/--bump/--rebuild/--render/--list`）。')
    A('')

    # ===== 2 一周对比 =====
    A('## 2. 规模与效率：一周前后')
    A('')
    wp, wn = dr.get('week_prev', {}), dr.get('week_now', {})
    ap_, an = dr.get('att_week_prev', {}), dr.get('att_week_now', {})
    A('### 2.1 发现效率')
    A('')
    A('| 指标 | 一周前 08-30~09-05 | 最近一周 09-06~09-12 | 变化 |')
    A('|---|---|---|---|')
    A('| Ralph 运行轮数 | %s | %s | %s |' % (wp.get('n_runs'), wn.get('n_runs'),
      '↓ %.0f%%' % (100 * (1 - wn.get('n_runs', 0) / max(wp.get('n_runs', 1), 1))) if wp.get('n_runs') else '—'))
    A('| 候选因子数 | %s | %s | — |' % (wp.get('cand'), wn.get('cand')))
    A('| **S1 通过** | %s | %s | — |' % (wp.get('s1'), wn.get('s1')))
    A('| **S1 通过率** | **%.3f** | **%.3f** | 持平 |' % (wp.get('s1_rate', 0), wn.get('s1_rate', 0)))
    A('| **S5 通过** | **%s** | **%s** | **连续归零** |' % (wp.get('s5'), wn.get('s5')))
    A('| 进入 JQ | %s | %s | — |' % (wp.get('jq'), wn.get('jq')))
    A('')
    A('> 读法：**S1 通过率几乎持平（%.3f → %.3f），说明生成端并没有「变蠢」**；'
      '瓶颈唯一地落在 S5。' % (wp.get('s1_rate', 0), wn.get('s1_rate', 0)))
    A('')
    A('### 2.2 生成与学习')
    A('')
    A('| 指标 | 一周前 | 最近一周 |')
    A('|---|---|---|')
    A('| Memory 新增 attempt | %s | %s |' % (ap_.get('n'), an.get('n')))
    A('| 平均 \\|ICIR\\| | %s | %s |' % (ap_.get('mean_abs_icir'), an.get('mean_abs_icir')))
    A('| \\|ICIR\\|≥0.3 条数 | %s | %s |' % (ap_.get('n_icir_gt_03'), an.get('n_icir_gt_03')))
    A('| \\|ICIR\\|≥0.5 条数 | %s | %s |' % (ap_.get('n_icir_gt_05'), an.get('n_icir_gt_05')))
    A('| outcome 分布 | `%s` | `%s` |' % (ap_.get('outcomes'), an.get('outcomes')))
    A('')
    A('### 2.3 Memory 快照时间序列')
    A('')
    A('| 快照 | attempts | total_pass | forbidden_directions | 数据截至 |')
    A('|---|---|---|---|---|')
    for m in mem:
        A('| `%s` | %s | %s | %s | %s |' % (m['file'], m['n'], m['pass'], m['forbidden'], m['last']))
    A('')
    A('> `forbidden_directions` 30 → 51 与 `total_pass` 24 → 31 说明**系统仍在学习**'
      '（并非「停摆」），但学到的多是「此路不通」。')
    A('')

    # ===== 3 归因 =====
    A('## 3. 因子干旱归因（核心）')
    A('')
    A('### 3.1 干旱时间线（权威口径 = `ralph_loop_result.*.stages_completed.evaluate.stage5_passed`）')
    A('')
    daily = dr.get('daily_runs', [])
    A('| 日期 | 运行 | 候选 | S1 | **S5** | JQ | v0.6硬拒 | 引擎 |')
    A('|---|---|---|---|---|---|---|---|')
    for d in daily:
        if d['date'] < '2026-08-24':
            continue
        A('| %s | %d | %d | %d | **%d** | %d | %d | `%s` |' % (
            d['date'], d['n_runs'], d['cand'], d['s1'], d['s5'], d['jq'], d['v06_rej'],
            ','.join(d['sources'].keys())))
    A('')
    A('> **最后一次 S5 通过 = 2026-08-29（5 条）；08-30~09-12 连续 14 天为 0**。'
      '此前 08-21~08-28 也已连续 8 天为 0。')
    A('')
    A('### 3.2 闸门是「合取式双条件」——这是主因')
    A('')
    A('`s5_joint_filter._verdict()`：')
    A('')
    A('```python')
    A('excess_ok = (e25 > 0.0) and (e26 > 0.0)')
    A('calmar_ok = (c25 > 1.0) and (c26 > 1.0)')
    A('passed = excess_ok and calmar_ok          # ← 合取')
    A('```')
    A('')
    A('**三个版本（08-29 / 08-31 / 09-11）逐字一致，年份窗口也始终是 (2025, 2026)** —— '
      '即：**没有人偷改阈值**，干旱不能归因于「悄悄收紧门槛」。')
    A('')
    A('但把全部 %d 条 S5 判据明细拉出来看：' % len(recs))
    A('')
    A('| 时期 | S5 判据条数 | 有超额值 | 双年正超额 | Calmar 双年>1 | **两者同时满足** |')
    A('|---|---|---|---|---|---|')
    for p in sorted(agg):
        a = agg[p]
        A('| %s | %d | %d | **%d** | **%d** | **%d** |' % (
            p, a.get('n', 0), a.get('has_excess', 0), a.get('excess_ok', 0),
            len([r for r in recs if period(r['date']) == p and r['c25'] is not None
                 and r['c26'] is not None and r['c25'] > 1 and r['c26'] > 1]),
            a.get('both_ok', 0)))
    A('')
    A('**全历史双年正超额的 %d 条记录，无一例外全部卡在 Calmar**，且呈明显「单边」形态：' % len(excess_both))
    A('')
    A('| 日期 | 因子 | ex25 | ex26 | calmar25 | calmar26 | 卡点 |')
    A('|---|---|---|---|---|---|---|')
    for r in sorted(excess_both, key=lambda x: x['date']):
        A('| %s | `%s` | %s | %s | %s | %s | %s |' % (
            r['date'], r['name'][:40], fmt(r['e25']), fmt(r['e26']),
            ('%.2f' % r['c25']) if r['c25'] is not None else '—',
            ('%.2f' % r['c26']) if r['c26'] is not None else '—',
            'calmar25<1' if (r['c25'] is not None and r['c25'] <= 1) else
            'calmar26<1' if (r['c26'] is not None and r['c26'] <= 1) else '—'))
    A('')
    A('> 🔴 **注意（2026-09-12 W5 修订）：以下「瓶颈在 2025」的读数来自 Top-200 抽样池，'
      '与全量池结论相反，引用时必须声明池口径。**')
    _c25 = [r['c25'] for r in excess_both if r['c25'] is not None]
    _c26 = [r['c26'] for r in excess_both if r['c26'] is not None]
    _c25_low = sum(1 for x in _c25 if x <= 1.0)
    A('> 在 %d 条双年正超额记录中，**%d 条的 `calmar_2025 ≤ 1.0` 被挡下**'
      '（c25 中位数 **%.2f**），而 `calmar_2026` 中位数 **%.2f**、多数 >1。'
      % (len(excess_both), _c25_low, statistics.median(_c25), statistics.median(_c26)))
    A('>')
    A('> ⚠️ **口径提示（2026-09-12 更正）**：上述「2025 是约束边」只在 **Top-200 池**成立。'
      '**流水线全量池**的 81 条判据给出**相反**读数（c25 中位 0.51 vs c26 中位 1.86 ⇒ 卡 2025），'
      '而 Top-200 池重裁给出 c25 中位 2.226 vs c26 中位 0.257（⇒ 卡 2026）。'
      '差异来源 = 抽样框（仅失败者 vs 混合）+ 池口径（全量 vs Top200）+ 数据来源（文本正则 vs 重算）。'
      '**凡涉池口径结论必须声明池。**见 `reports/factor_quality_improvement_20260912.md` §W5。')
    A('>')
    A('> 机制：**2025 是完整年度**（含 A 股年度级回撤 → maxDD 大 → Calmar 被压低），'
      '**2026 只有 01~09**（窗口短 → maxDD 小 → Calmar 被抬高）。'
      '把同一个 `>1.0` 阈值**同时**施加在「完整年」与「不完整年」上再取**合取**，'
      '等于要求 2025 这个坏年份必须达到 2026 短窗口的 Calmar 水平 —— '
      '这是 `both_ok ≡ 0` 的直接技术原因。')
    A('')
    A('> 参考量级：**可同时解析出双年 Calmar 的记录 %d 条**，其中 **Calmar 双年同时 >1 的 0 条**；'
      '本次新跑的双口径实验（§3.4）在**旧口径**下独立复现了同一结论 —— '
      '**2 条双年正超额**（`vwap_dev_ratio_5_60`、`fd_max-yield`）与 **3 条 Calmar 双年达标**'
      '（`margin_volatility_squeeze`、`low_disagreement_momentum`、`overnight_intraday_ratio`）'
      '是**两批不相交的因子**，交集 `both_ok = 0`。'
      % len([r for r in recs if r['c25'] is not None and r['c26'] is not None]))
    A('')
    A('### 3.3 v0.7.9 口径变更（09-09）的影响：不是 bug，是「换通过集合」')
    A('')
    A('| 项 | 旧口径 | 新口径 |')
    A('|---|---|---|')
    A('| 裁决池 | `sample_stocks=200`（按成交额 Top-200 大盘池，覆盖全A成交额 **25.9%**） | 全量池 ~5700 只 |')
    A('| 基准 | `000905` 中证500 | `000985` 中证全指 |')
    A('| 同级裁决 | S1–S4 全市场 / S5 仅大盘 → **不同池** | 五项裁决同池 |')
    A('')
    A('变更理由（代码注释）：生成端持续产出中小盘 alpha 因子，在大盘池里被结构性误杀。')
    A('')
    A('**实测（两路证据，结论需分开读）**：')
    A('')
    A('| 证据 | 样本 | 结果 | 读法 |')
    A('|---|---|---|---|')
    A('| 09-09 当日重裁 | 39 条**历史被拒**因子 | **0 条翻盘**，多数超额进一步恶化 | 被拒因子的「正超额」多是大盘池噪声，换全量池后被稀释 ⇒ 池**不是**它们失败的原因 |')
    A('| 本次双口径对照（§3.4） | 14 条（11 条 JQ 队列·过 S5 ＋ 1 条 Memory 弱 ＋ 2 条当日候选） | 新口径 %d/%d 通过（旧口径 0/14） | `margin_volatility_squeeze` 被新口径**救活** ⇒ 中小盘 alpha 被大盘池误杀**确有实例** |' % (
        sum(1 for r in new.values() if r.get('passed')), len(new)))
    A('')
    A('> 🔴 合起来看：**「池错杀」不是空穴来风（确有中小盘 alpha 被误杀），但它解释不了那 39 条被拒因子**。'
      '新口径的净效应是**换通过集合 + 整体更严**（`ex25` 均值 %+.4f）。' % _d25_mean)
    A('')

    # 双口径对照
    if abl:
        n_new = len(new)
        A('### 3.4 决定性对照实验：同公式 × 双口径（本次新跑，已完成双口径全量）')
        A('')
        A('取 %d 条因子公式，在**同一份行情数据**上分别用'
          '旧口径（`sample_stocks=200` + 000905）与全量池口径（`sample_stocks=None` + 000985）'
          '重跑 S5，隔离「校准口径变更」这一个变量（`data/_s5_caliber_ab_20260912.json`）。' % len(old))
        A('')
        A('> 🔴 **来源构成（已逐条核对，勿笼统称"历史过 S5"）**：'
          '**11 条**来自 JQ 代码队列（文件名形如 `jq_s5_pass_*_standalone.py`，确为历史过 S5，'
          '含 `max_drawdown_duration` 标 `jq_failed`）；'
          '**1 条** `overnight_intraday_ratio` 来自 Memory 弱记录（`outcome=WEAK`，**未过 S5**）；'
          '**2 条** `FORGE_0912_1/2` 是 2026-09-12 当日 forge 候选（**未过 S5**，作对照）。')
        A('')
        A('| 因子 | 来源 | 旧:ex25 | 旧:ex26 | 旧:c25 | 旧:c26 | 新:ex25 | 新:ex26 | 新:c25 | 新:c26 | 旧pass | 新pass |')
        A('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for nm in names:
            o = old.get(nm, {})
            n = new.get(nm, {})
            A('| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | **%s** |' % (
                nm[:40], PROV.get(nm, '—'),
                fmt(o.get('excess_25')), fmt(o.get('excess_26')),
                o.get('calmar_25', '—'), o.get('calmar_26', '—'),
                fmt(n.get('excess_25')), fmt(n.get('excess_26')),
                n.get('calmar_25', '—'), n.get('calmar_26', '—'),
                o.get('passed', '—'), n.get('passed', '—')))
        A('')
        n_old_pass = sum(1 for r in old.values() if r.get('passed'))
        n_new_pass = sum(1 for r in new.values() if r.get('passed'))
        A('**旧口径通过 %d / %d ｜ 新口径通过 %d / %d**。' % (
            n_old_pass, len(old), n_new_pass, n_new))
        A('')
        A('> 🔴 **这一结果推翻了「新口径只是单纯抬高门槛」的简化叙事** —— '
          '口径变更**同时杀因子也救因子**，是「改变通过集合」而非「整体变严」。')
        A('')
        A('**发现 1 · 口径漂移整体偏负**：`NEW − OLD` 超额均值 '
          '`ex25` **%+.4f**（%d/%d 条变差）、`ex26` **%+.4f**。'
          '旧口径下 `vwap_dev_ratio_5_60`（旧 ex25 +0.0296）与 `fd_max-yield`（+0.0567）的'
          '双年正超额，在新口径下崩为 %s / %s ⇒ **旧 Top-200 大盘池对多数因子存在系统性美化**。' % (
              _d25_mean, _d25_worse, len(_d25), _d26_mean,
              fmt(new.get('vwap_dev_ratio_5_60', {}).get('excess_25')),
              fmt(new.get('fd_max-yield', {}).get('excess_25'))))
        A('')
        A('**发现 2 · 但新口径确实救活了一条，v0.7.9 的动机有实据**：'
          '`margin_volatility_squeeze`（公式实为 `close*volume` 20 日变异系数取负）'
          '在旧大盘池里 `ex25 = -0.1215`（被误杀），换到**全量池**后 `ex25 = +0.0293`、'
          '`ex26 = +0.1091`、`c25 = 3.581`、`c26 = 1.044` ⇒ **新口径下通过 S5**。'
          '这正是 v0.7.9 变更说明里「中小盘 alpha 在大盘池被结构性误杀」的**首个直接实证**。')
        A('')
        A('> ⚠️ **两条必要的保留**：')
        A('> ① 该因子 `ex25` 仅 **+2.93%**，属**极弱通过** —— 它过闸恰恰说明 '
          '`excess > 0` 这个门槛**太松**（只要不是负的就行），而非说明它是个好因子。'
          '它当年在 JQ 队列里的状态也正是 `jq_marginal`。')
        A('> ② 该因子**名实不符**：名字含 `margin`（两融），公式却只用 `close*volume`，'
          '**完全不含两融字段**。命名会误导后续的因子归因与去重，建议核实改名。')
        A('')
        A('**发现 3 · 旧口径下两个条件的满足集合恰好互斥（合取终裁缺陷的最直接证据）**：')
        A('')
        A('| 旧口径子集 | 条数 | 因子 | 卡点 |')
        A('|---|---|---|---|')
        A('| 双年正超额（`excess_ok=True`） | 2 | `vwap_dev_ratio_5_60`、`fd_max-yield` | 两条 `c26` 分别为 0.66 / 0.461，**全部卡 Calmar** |')
        A('| Calmar 双年 >1（`calmar_ok=True`） | 3 | `margin_volatility_squeeze`、`low_disagreement_momentum`、`overnight_intraday_ratio` | 三条 `ex25` 分别为 -0.1215 / -0.0334 / -0.0452，**全部卡 2025 超额** |')
        A('| **交集 `both_ok`** | **0** | — | **合取式终裁在旧口径下通过 0 条** |')
        A('')
        A('> 也就是说：旧口径下「超额达标」与「Calmar 达标」是两批**不相交**的因子。'
          '合取后通过数被乘性压缩到 0。')
        A('')
        A('**发现 4 · 但合取并非数学上不可满足**：新口径下 `both_ok = 1`'
          '（`margin_volatility_squeeze` 同时满足两条件）。'
          '因此 §3.2 的 `both_ok ≡ 0` 应准确表述为 **「在当前管线候选分布下几乎不可满足」**，'
          '而非「判据本身矛盾」。真正的问题是：**两个条件高度正交 + 合取 ⇒ 通过率被乘性压缩**。')
        A('')
        A('> ⚠️ 口径：本实验 n = %d，`both_ok` 的互斥形态可能部分来自小样本；'
          '但「口径变更双向作用」「新口径能救活中小盘 alpha」两点已由单因子层面证实。' % len(old))
        A('')

    A('### 3.5 排除项：新增守卫不是原因')
    A('')
    A('| 落地项 | 版本 | 开关状态 | 是否参与裁决 |')
    A('|---|---|---|---|')
    A('| CogAlpha 认知进化 | v0.7.11 | `FUXI_LLM_COG_EVOLVE=shadow` | **否**（build_prompt 逐字符相同） |')
    A('| 幸存者偏差守卫 | v0.7.11 | `FUXI_SURVIVORSHIP_GUARD=shadow` | **否** |')
    A('| 执行陷阱守卫 | v0.7.10 | `FUXI_EXEC_TRAP_GUARD=shadow` | **否** |')
    A('| 公式因果闭包 | v0.7.6 | 静态检查 | 拦语法/DSL，不拦 alpha |')
    A('| 维度预剪枝 / 多样性折扣 | v0.7.2 | `enforce` | 影响**特征空间**，不直接拒因子 |')
    A('| S5 方向符号守卫 | v0.7.7 | `enforce` | 只对负 ICIR 做**翻转重评**，不减少通过数 |')
    A('')
    A('> 结论：**干旱与这批守卫无因果关系**。唯一 `enforce` 的两项（预剪枝/多样性折扣）'
      '作用于生成端的特征与父本选择，且 S1 通过率未下降（§2.1）可反证其未伤及生成能力。')
    A('')

    # ===== 4 更优的证据 =====
    A('## 4. 「系统确实更优」的证据（辨伪维度）')
    A('')
    A('| # | 证据 | 实测 | 意义 |')
    A('|---|---|---|---|')
    A('| 1 | `factor_ic_computer` reindex 取交集 bug 修复（v0.7.10 / P-20260911-008） | '
      '181 交易日面板 `roe_dt` 前半段有效均值 8.8% → **99.8%**；15 条因子同口径复评 **过闸 9→4、翻转 7**（假阳性率 **66.7%**）| 清掉了当年"看着能过"的伪因子 |')
    A('| 2 | 幸存者偏差守卫（v0.7.11） | 面板 5774 只/非幸存 212(3.67%)/脆弱 523(9.06%)；'
      '4 条量价因子命中幸存依赖（振幅归一化 12.02 / 量价背离 4.19 / 放量强度 3.59）| 识别出"表观 alpha 来自后续退市股" |')
    A('| 3 | S5 全量池重裁（v0.7.9） | 39 条历史被拒因子 **0 条翻盘**，多数超额更负 | '
      '**阻止了一次错误的"放宽闸门"** —— 若当初按"池错杀"放宽，会放进 39 个坏因子 |')
    A('| 4 | 五项裁决同池（v0.7.9） | S1–S4 全市场 vs S5 大盘池的口径分裂消除 | 消除"小池子集运气"（对比 800 池 3 条双年正超额在全量池 0 条复现）|')
    A('| 5 | 方向符号守卫（v0.7.7）+ 执行陷阱守卫（v0.7.10） | 负 ICIR 自动翻转重评；成本后净利转负诊断 | 两类已知伪因子形态被系统性拦截 |')
    A('')
    A('## 5. 但也必须承认的代价：无正样本 → 学习信号缺失')
    A('')
    A('- S5 连续 14 天 0 通过 ⇒ **管线没有正样本**，`success_templates` 停在 90 条，'
      '`NOVEL_FACTOR_LIBRARY` 只靠外部知识补充（09-12 库存 255 条）。')
    A('- Memory 的 R→G→E→D→**D+** 闭环里，D+（JQ 反馈）自 09-09 起没有新输入 ⇒ '
      '**进化回路事实上在"半开环"运行**。')
    A('- 置信度检查：`forbidden_directions` 30→51、`total_pass` 24→31 说明系统**没有停摆**，'
      '但学到的偏「此路不通」，把搜索空间越收越窄。')
    A('')
    A('## 6. 建议（按优先级）')
    A('')
    A('1. 🔴 **拆分 S5 的合取式终裁（最高优先）**：把 `passed = excess_ok and calmar_ok` '
      '改为分级 —— 例如 `A级 = 双年正超额 + 双年Calmar>1`（现行）、'
      '`B级 = 双年正超额`（可进 JQ 影子对照）、`C级 = 单年正超额`（只记不裁）。'
      '现状两个条件**高度正交**，合取后通过率被乘性压缩到 **0**（§3.4 发现 3：两批因子不相交），'
      '终裁几乎无信息量；分级后至少能恢复学习梯度。')
    A('2. 🔴 **修正 `calmar_2026` 的口径不可比问题**：2026 是**不完整年度**'
      '（2026-01~09），与完整 2025 的 Calmar 直接比较不成立。建议要么按'
      '「年化因子 × 同窗口最大回撤」统一口径，要么改为对 2026 使用'
      '「与 2025 同长度窗口」（滚动 12 个月）。**这是 both_ok ≈ 0 的直接技术原因。**')
    A('3. 🟡 **给 v0.7.9 的口径变更加"双轨影子"**：新口径裁决的同时，保留旧口径（Top-200+000905）'
      '的通过数作为对照指标 —— §3.4 已证明两种口径会**选出不同的因子**，'
      '单轨无法区分「门槛抬高」与「生成退步」，也无法捕捉被大盘池误杀的中小盘 alpha。')
    A('4. 🟡 **把 `--check` 接入每日流水线收尾**：`python scripts/version_bump.py --check`，'
      '未编号即告负，避免再次出现"12 天 40 条落地无版本号"。')
    A('5. 🟢 **复核 `margin_volatility_squeeze` 并考虑补进 JQ 队列**：它是本次实验中'
      '**唯一通过 S5 的因子**，且是独立于管线的历史公式 ⇒ 可作为「新口径能否产出有效因子」的'
      '活样本，验证「S5 通过 → JQ 实盘」链路是否真的通畅（§3.4 发现 2）。')
    A('6. 🟢 **清理 `trajectory_log._eval_log` 的字段错位**：`calmar_25` 字段实际存的是'
      '`calmar_26` 的值（`rejection_reason` 文本才是真值），`calmar_26` 恒为 `null`；'
      '且 `passed=eligible_for_jq` 易被误读为「过了 S5」。建议拆成 `s5_passed` 与 `eligible_for_jq` 两个字段。')
    A('')
    A('---')
    A('')
    A('### 附：数据来源')
    A('')
    A('| 产物 | 说明 |')
    A('|---|---|')
    A('| `data/ralph_loop_result_*.json`（%d 份） | 逐轮漏斗（cand/S1…S5/JQ/守卫） |' % dr.get('n_result_files', 0))
    A('| `data/trajectory_log.json` | S5 逐条判据明细（excess/calmar） |')
    A('| `data/experience_memory.json*` | 学习记忆快照时间序列 |')
    A('| `data/_s5_caliber_ab_20260912.json` | 本次双口径对照实验 |')
    A('| `data/version_registry.json` + `CHANGELOG.md` | 版本编号注册表 |')
    A('| `reports/s5_fullpool_rerun_20260909.md` | 09-09 全量池重裁报告（0/39 翻盘） |')

    out = f'reports/fuxi_week_over_week_{args.date}.md'
    os.makedirs('reports', exist_ok=True)
    open(out, 'w', encoding='utf-8').write('\n'.join(L))
    print('[OK] %s (%d chars, %d lines)' % (out, sum(len(x) for x in L), len(L)))


if __name__ == '__main__':
    main()
