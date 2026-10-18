# -*- coding: utf-8 -*-
"""_bj_uni_j4b_replay_20261006.py — J4 负对照批次 2（只读）

背景: 批次 1 筛查发现零 .BJ 暴露的挑选规律 —— 公式含**成交量分母 / 标度标准化**
时, 北交所(920xxx.BJ, 新三板期成交量极小)天然排不进 TOP-80。批次 1 已命中
`gp_evo_002` (bj=0)。本批按同族扩筛, 目标是凑够 J4 要求的 ≥3 个合格负对照。

批次 2 候选 (全部标度无关族; 括号内为历史 S5 读数是否非 null):
  gp_evo_005   区间/量        (ex 非 null)
  gp_breed_039 区间/(区间+log量) (ex null —— 纳入以扩面, 以实测是否早退为准)
  forge_gen2   开盘 z-score 链   (ex 非 null)
  gp_evo_009   区间/相关链      (ex 非 null)

复用批次 1 的流程: SCREEN(legacy) → 取 ran ∧ 零 .BJ 暴露者 → PAIR(listed_only) → 对拍。
输出: reports/fuxi_v1_bj_universe_20261006/j4b_readings.json
"""
import importlib.util
import os

ROOT = r'E:\quant'
spec = importlib.util.spec_from_file_location(
    'j4mod', os.path.join(ROOT, 'scripts', '_bj_uni_j4_replay_20261006.py'))
j4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j4)

j4.CANDIDATES = [
    dict(name='gp_evo_005', tag='区间/量',
         formula='(-((high - amount_p) / (volume.rolling(20).mean() + 1)).rank(pct=True))',
         freq='daily'),
    dict(name='gp_breed_039', tag='区间/(区间+log量)',
         formula='((high - close) / ((high - low) + np.log(volume_p))).rank(pct=True)',
         freq='daily'),
    dict(name='forge_gen2_open_zchain', tag='开盘z-score链',
         formula='(1.0 / (((open_p - open_p.rolling(12).mean()) / '
                 '(open_p.rolling(12).std() + 1e-6)) + 1e-6)).rolling(20).std()'
                 '.rolling(12).mean().rolling(5).sum()', freq='daily'),
    dict(name='gp_evo_009', tag='区间/相关链',
         formula='(-((high - low) / ((-(-((hl_ratio - hl_ratio.shift(5)) - '
                 'high_p.rolling(15).cov(close_p)))).mean() + 1)).rank(pct=True))',
         freq='daily'),
]

j4.READINGS = os.path.join(j4.OUT, 'j4b_readings.json')

if __name__ == '__main__':
    j4.main()
