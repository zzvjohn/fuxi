# -*- coding: utf-8 -*-
"""P2-1c 词汇表反漂移单测 (v0.7.108, 2026-09-28)

目标: 断言四套门禁白名单与 S5 求值面的一致性, 防同族缺陷第 4 次复发
(ewm v0.6.2 → moneyflow 半个 → median + lhb_tri 9 + _p 变体 v0.7.108)。

求值面定义 (真相源: s5_joint_filter.py 面板构建/注入段, 行号在 9-28 检查时核对):
  base 7 (1125-1132 含 pre_close) + derived 5 (1134-1144)
  moneyflow 10 (1219-1241)  | margin 8 (_MARGIN_COLS, 1262-1277)
  balancesheet 3 (1288-1354) | valuation 3 + fina_quality 3 (1386-1482)
  top_list 4 (1454-1482)     | top_inst_tri 9 (_LHB_TRI_COLS, 1518-1541)
  fundamental_quarter roe/eps/eps_qoq (2064+ fund_globals, 截面路径专属)
全部字段均按 `df[f'{col}_p'] = df[col]` 模式带 _p 副本 (fund 三项除外)。

运行: cd research/factor_alchemy && python -m unittest tests.test_vocab_alignment_20260928 -v
"""
import ast
import re
import unittest
from pathlib import Path

FA = Path(__file__).resolve().parent.parent
S5 = FA / 's5_joint_filter.py'
FET = FA / 'factor_expression_tree.py'
FQG = FA / 'factor_quality_gate.py'

# 非因子内部列 (df[...] 注入但不是公式字段): 护栏扫描豁免
INTERNAL_COLS = {'factor_value', 'idx_return', 'trade_date', 'ts_code'}
# ts_pct 豁免: DSL 算子, 翻译层应转 pct_change; 残余形态保持被拒=保守
EVAL_KEYWORD_EXEMPT = {'ts_pct'}


def _both(names):
    out = []
    for n in names:
        out += [n, n + '_p']
    return out


def _extract_consts(tree):
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and tg.id in ('_LHB_TRI_COLS', '_MARGIN_COLS'):
                    consts[tg.id] = [e.value for e in node.value.elts]
    return consts


def _set_literal(path, name):
    t = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in ast.walk(t):
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and tg.id == name and isinstance(node.value, ast.Set):
                    return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return None


def _func_local_set(path, func_name, var_name):
    t = ast.parse(Path(path).read_text(encoding='utf-8'))
    consts = _extract_consts(t)
    for node in ast.walk(t):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for tg in sub.targets:
                        if isinstance(tg, ast.Name) and tg.id == var_name and isinstance(sub.value, ast.Set):
                            lit = {e.value for e in sub.value.elts if isinstance(e, ast.Constant)}
                            for e in sub.value.elts:
                                if isinstance(e, ast.Starred):
                                    if isinstance(e.value, ast.Name) and e.value.id == '_MARGIN_COLS':
                                        lit |= set(consts['_MARGIN_COLS'])
                                    elif isinstance(e.value, ast.ListComp):
                                        lit |= {c + '_p' for c in consts['_MARGIN_COLS']}
                            return lit
    return None


class TestVocabAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t5 = ast.parse(S5.read_text(encoding='utf-8'))
        cls.consts = _extract_consts(t5)
        cls.face = set(
            _both(['open', 'high', 'low', 'close', 'volume', 'amount', 'pre_close']) +
            _both(['overnight', 'amplitude', 'returns', 'turnover', 'hl_ratio']) +
            _both(['buy_lg_vol', 'sell_lg_vol', 'buy_sm_vol', 'sell_sm_vol',
                   'buy_md_vol', 'sell_md_vol', 'buy_elg_vol', 'sell_elg_vol',
                   'net_mf_vol', 'net_mf_amount']) +
            _both(cls.consts['_MARGIN_COLS']) +
            _both(['intan_assets', 'goodwill', 'total_assets']) +
            _both(['pe_ttm', 'pb', 'dv_ratio']) +
            _both(['roe_dt', 'tr_yoy', 'netprofit_yoy']) +
            _both(['lhb_flag', 'lhb_net_amount', 'lhb_amount', 'lhb_net_rate']) +
            _both(cls.consts['_LHB_TRI_COLS']) +
            ['roe', 'eps', 'eps_qoq']
        )
        cls.cross_only = {'roe', 'eps', 'eps_qoq'}
        cls.per_stock = cls.face - cls.cross_only
        cls.G1 = _set_literal(FET, 'S5_AVAILABLE_FIELDS')
        cls.G2 = _set_literal(FET, 'S5_EVAL_KEYWORDS')
        cls.G3ops = _set_literal(FQG, 'KNOWN_OPS')
        cls.G3fields = _set_literal(FQG, 'KNOWN_FIELDS')
        cls.G4 = _func_local_set(S5, '_check_unknown_fields', 'known')
        # field_cols 面
        fc = None
        for node in ast.walk(t5):
            if isinstance(node, ast.FunctionDef) and node.name == '_build_series_context':
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for tg in sub.targets:
                            if isinstance(tg, ast.Name) and tg.id == 'field_cols' and isinstance(sub.value, ast.List):
                                fc = {e.value for e in sub.value.elts if isinstance(e, ast.Constant)}
        cls.field_cols = set(fc or [])
        cls.field_cols |= set(cls.consts['_LHB_TRI_COLS']) | {c + '_p' for c in cls.consts['_LHB_TRI_COLS']}
        cls.field_cols |= {'close', 'open', 'high', 'low', 'volume', 'amount'}

    def test_face_subset_s5_available_fields(self):
        """GP 硬拒门 (S5_AVAILABLE_FIELDS) 必须覆盖求值面 (v0.7.108 修复点)"""
        missing = self.face - self.G1
        self.assertEqual(set(), missing, f'GP 门禁误拒面: {sorted(missing)}')

    def test_face_subset_known_fields(self):
        """质量门 (KNOWN_FIELDS) 必须覆盖求值面"""
        missing = self.face - self.G3fields
        self.assertEqual(set(), missing, f'质量门误拒面: {sorted(missing)}')

    def test_face_subset_s5_known(self):
        """S5 层 _check_unknown_fields.known 必须覆盖求值面"""
        missing = self.face - self.G4
        self.assertEqual(set(), missing, f'S5 层 unknown 面: {sorted(missing)}')

    def test_known_ops_subset_eval_keywords(self):
        """质量门算子集 ⊆ GP 关键词集 (ts_pct 显式豁免, 见 EVAL_KEYWORD_EXEMPT)"""
        missing = self.G3ops - self.G2 - EVAL_KEYWORD_EXEMPT
        self.assertEqual(set(), missing, f'算子对齐缺口: {sorted(missing)}')

    def test_median_present_in_all_operator_sets(self):
        """median 三处齐备 (P2-1 实证误拒点, 防回归)"""
        self.assertIn('median', self.G2)
        self.assertIn('median', self.G3ops)

    def test_field_cols_covers_per_stock_face(self):
        """per-stock field_cols 面 ⊇ per-stock 求值面 (双向一致)"""
        missing = self.per_stock - self.field_cols
        self.assertEqual(set(), missing, f'field_cols 缺注入: {sorted(missing)}')

    def test_no_orphan_df_columns(self):
        """护栏: df["literal"] 注入键 ⊆ 求值面 ∪ 内部列 (防新数据源漏进白名单)"""
        src = S5.read_text(encoding='utf-8')
        keys = set(re.findall(r'df\["([a-zA-Z_][a-zA-Z0-9_]*)"\]', src))
        keys |= set(re.findall(r"df\['([a-zA-Z_][a-zA-Z0-9_]*)'\]", src))
        orphan = sorted(keys - self.face - INTERNAL_COLS)
        self.assertEqual([], orphan, f'未登记的面板注入列: {orphan}')

    def test_ts_pct_exemption_still_stands(self):
        """豁免留痕: ts_pct 不在 G2; 若加入则本条失败, 迫使删除豁免记录"""
        for exempt in EVAL_KEYWORD_EXEMPT:
            self.assertNotIn(exempt, self.G2, f'{exempt} 已加入 G2, 应删除 EVAL_KEYWORD_EXEMPT 豁免')

    def test_no_reverse_orphans_in_gp_gate(self):
        """反向护栏: G1 不得含无法在求值面兑现的残留名 (v0.7.108 后为零)
        注: KNOWN_FIELDS 的 19 个历史名 (atr/vwap/market_cap/...) 不在断言范围 —
        它们是 FactorICComputer 时代遗产, 属另一个轴, 见报告登记。"""
        reverse = self.G1 - self.face
        self.assertEqual(set(), reverse, f'G1 含求值面外名字: {sorted(reverse)}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
