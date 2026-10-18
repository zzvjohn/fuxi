# -*- coding: utf-8 -*-
"""补丁: s5_joint_filter.py 轴语义/字段漂移/静默丢候选 四处修复 (2026-09-18, P-20260918-006)

A. cross-section pivot 字段清单**公式驱动自动补全**
   原硬编码清单与求值器字段集合漂移 ⇒ 公式引用未 pivot 的列时 NameError
   → 静默丢候选 (实锤: novel_234~240 三路席位列, 7 条)。
B. _needs_cross_section 字面子串白名单 → 加**正则**兜底 (容忍空白/参数顺序)
   原写法 `rank(pct=True,axis=1)` 无空格 / 插入 ascending= 均漏判 (实测 21 条潜在)。
   只加不减 ⇒ 不改变当前能通过公式的路由。
C. 逐股路径无结果 + 公式含 axis 记号 ⇒ 升级到截面路径重试一次 (**留痕**)
   安全网: 即使路由漏判也不静默丢候选; 只在 Series 路径已失败时触发 = 零行为变化。
D. 截面求值产出**全 NaN** 时打印明确诊断 (原为完全静默)
   成因: DataFrame(日期×股票) 与 Series(索引=日期) 直接加减, pandas 按**列**对齐
   ⇒ 全 NaN; 应按 axis=0 对齐。实测 5 条因子 (novel_040/070/073/099/203) 因此恒死。

CRLF 保真: 读 → LF 归一 → 替换 → 整体写回 CRLF。
"""
import py_compile
import shutil

P = r'e:\quant\research\factor_alchemy\s5_joint_filter.py'

# ── A1: 模块级正则 ────────────────────────────────────────────────
OLD_A1 = "def _lhb_tri_on() -> bool:"
NEW_A1 = '''# 2026-09-18 (P-20260918-006): 任一 pandas 调用带 axis=1 的正则判据。
# 容忍任意空白与参数顺序 (`rank(axis=1, pct=True)` / `rank(pct=True,axis=1)` /
# `rank(pct=True, ascending=False, axis=1)` 全命中), 用于补齐字面白名单的漏判。
_AXIS1_CALL_RE = re.compile(r'\\.\\s*\\w+\\s*\\([^()]*\\baxis\\s*=\\s*1\\b')


def _lhb_tri_on() -> bool:'''

# ── B: _needs_cross_section 加正则兜底 ────────────────────────────
OLD_B = """        has_axis1 = any(m in formula for m in ['mean(axis=1)', 'mean(axis = 1)',
                                               'std(axis=1)', 'std(axis = 1)',
                                               'min(axis=1)', 'min(axis = 1)',
                                               'max(axis=1)', 'max(axis = 1)',
                                               'rank(axis=1', 'rank(axis = 1',
                                               'rank(pct=True, axis=1'])"""
NEW_B = """        has_axis1 = any(m in formula for m in ['mean(axis=1)', 'mean(axis = 1)',
                                               'std(axis=1)', 'std(axis = 1)',
                                               'min(axis=1)', 'min(axis = 1)',
                                               'max(axis=1)', 'max(axis = 1)',
                                               'rank(axis=1', 'rank(axis = 1',
                                               'rank(pct=True, axis=1'])
        # 2026-09-18 (P-20260918-006): 字面白名单只覆盖"恰好这么写"的形式;
        # 写法变体 (逗号后无空格 / 插入 ascending= / 参数顺序对调) 会漏判 ⇒
        # 漏判后落进逐股 Series 上下文 ⇒ Series.rank(axis=1) ValueError ⇒ 静默丢。
        # 实测: novel_217~226 改 `rank(pct=True,axis=1)` 后路由判定 True→False。
        # 此兜底**只加不减**, 不改变当前能通过公式的路由。
        if not has_axis1:
            has_axis1 = bool(_AXIS1_CALL_RE.search(formula))"""

# ── A2: pivot 字段清单公式驱动补全 ────────────────────────────────
OLD_A2 = """                          'lhb_flag', 'lhb_net_amount', 'lhb_amount', 'lhb_net_rate']:
            if extra_col in df.columns:
                fields.append(extra_col)
        available = [f for f in fields if f in df.columns]"""
NEW_A2 = """                          'lhb_flag', 'lhb_net_amount', 'lhb_amount', 'lhb_net_rate']:
            if extra_col in df.columns:
                fields.append(extra_col)
        # ── 2026-09-18 (P-20260918-006): 公式驱动字段补全 ──────────────
        # 上面是硬编码清单, 与求值侧字段集合会漂移: 公式引用了清单外的列时
        # cross-section 路径 NameError → **静默丢候选**。
        # 实锤: novel_234~240 (三路龙虎榜席位因子) 因 lhb_{inst,north,yz}_* 未进
        # 清单, 在截面路径全军覆没 (逐股路径已接线, 二者不同步)。
        # 修法: 从公式抽取标识符 ∩ df.columns 补入 fields。成本有界 (随公式长度),
        # 不采用"pivot 全部列"以避免本项目 2.48GiB 级内存前科复现。
        for _tok in sorted(set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', formula))):
            if _tok in df.columns and _tok not in fields:
                fields.append(_tok)
        available = [f for f in fields if f in df.columns]"""

# ── D: 全 NaN 诊断 ────────────────────────────────────────────────
OLD_D = """            if isinstance(result, pd.DataFrame):
                # 将宽表转回长表
                result_long = result.stack().reset_index()
                result_long.columns = ['trade_date', 'ts_code', 'factor_value']
                result_long = result_long.dropna(subset=['factor_value'])
                result_long = result_long[np.isfinite(result_long['factor_value'])]
                return result_long"""
NEW_D = """            if isinstance(result, pd.DataFrame):
                # 将宽表转回长表
                result_long = result.stack().reset_index()
                result_long.columns = ['trade_date', 'ts_code', 'factor_value']
                _n_raw = len(result_long)
                result_long = result_long.dropna(subset=['factor_value'])
                result_long = result_long[np.isfinite(result_long['factor_value'])]
                # 2026-09-18 (P-20260918-006): 原实现在此**完全静默** —
                # eval 成功但整表 NaN 时直接返回空表, 调用方无从区分
                # "公式写错" vs "真的没信号"。补明确诊断 (留痕), 不改变返回值。
                if len(result_long) == 0 and _n_raw > 0:
                    print(f"    [S5] {factor_name}: 截面求值产出 {_n_raw} 格但**全为 "
                          f"NaN/Inf** ⇒ 静默丢候选。常见成因: DataFrame(日期×股票) 与 "
                          f"Series(索引=日期) 直接 `-`/`+`, pandas 按**列**对齐 ⇒ 全 NaN; "
                          f"应改 `.sub(Y, axis=0)` 按行对齐")
                return result_long"""
NEW_D = NEW_D.replace('`-`/`+`', '加减')

# ── C: 逐股 → 截面 升级安全网 ─────────────────────────────────────
OLD_C = """        try:
            if needs_cross_section:
                # ── 全面板模式: 构建多股票 DataFrame 上下文 ──
                return self._evaluate_cross_section(df, clean_formula, factor_name)
            else:
                # ── 单股票模式: 逐股 eval ──
                return self._evaluate_per_stock(df, clean_formula, factor_name)
        except Exception as e:
            print(f"    [S5] Error evaluating {factor_name}: {e}")
            return None"""
NEW_C = """        try:
            if needs_cross_section:
                # ── 全面板模式: 构建多股票 DataFrame 上下文 ──
                return self._evaluate_cross_section(df, clean_formula, factor_name)
            # ── 单股票模式: 逐股 eval ──
            _res = self._evaluate_per_stock(df, clean_formula, factor_name)
            # 2026-09-18 (P-20260918-006) 安全网: 路由判据无论多宽都可能漏。
            # 逐股路径**无结果**且公式含 axis 记号 ⇒ 升级到截面路径重试一次, 并留痕
            # (三态: 未走 / 升级后通 / 升级后仍空), 杜绝静默丢候选。
            # 只在 Series 路径已失败时触发 ⇒ 对当前能通过的公式**零行为变化**。
            if (_res is None or len(_res) == 0) and 'axis' in clean_formula:
                print(f"    [S5] {factor_name}: 逐股路径无结果且公式含 axis 记号 "
                      f"→ 升级到截面路径重试 (留痕)")
                _res2 = self._evaluate_cross_section(df, clean_formula, factor_name)
                if _res2 is not None and len(_res2):
                    print(f"    [S5] {factor_name}: 升级成功 ({len(_res2)} 行) — "
                          f"说明路由判据曾漏判, 请补正则")
                else:
                    print(f"    [S5] {factor_name}: 升级后仍无结果 (截面路径亦空/报错)")
                return _res2
            return _res
        except Exception as e:
            print(f"    [S5] Error evaluating {factor_name}: {e}")
            return None"""

PATCHES = [
    ('A1 模块级正则', OLD_A1, NEW_A1),
    ('B  _needs_cross_section 正则兜底', OLD_B, NEW_B),
    ('A2 pivot 字段公式驱动补全', OLD_A2, NEW_A2),
    ('D  全 NaN 诊断', OLD_D, NEW_D),
    ('C  逐股→截面 升级安全网', OLD_C, NEW_C),
]


def main():
    shutil.copy2(P, P + '.bak_20260918_axis')
    print('backup:', P.split('\\')[-1] + '.bak_20260918_axis')

    raw = open(P, 'rb').read()
    s = raw.decode('utf-8')
    n_crlf0 = s.count('\r\n')
    print('before: bytes=%d CRLF=%d bareLF=%d' % (len(raw), n_crlf0, s.count('\n') - n_crlf0))

    lf = s.replace('\r\n', '\n')
    for name, o, n in PATCHES:
        c = lf.count(o)
        status = 'OK ' if c == 1 else 'FAIL'
        print('  [%s] %-36s occurrences=%d' % (status, name, c))
        if c != 1:
            raise SystemExit('ABORT at %s: expected 1, got %d' % (name, c))
        lf = lf.replace(o, n, 1)

    out = lf.replace('\n', '\r\n')
    open(P, 'wb').write(out.encode('utf-8'))

    raw2 = open(P, 'rb').read()
    s2 = raw2.decode('utf-8')
    n_crlf = s2.count('\r\n')
    n_lf = s2.count('\n') - n_crlf
    print('after : bytes=%d CRLF=%d bareLF=%d' % (len(raw2), n_crlf, n_lf))
    assert n_lf == 0, 'bare LF introduced!'

    py_compile.compile(P, doraise=True)
    print('py_compile OK')

    for probe in ['_AXIS1_CALL_RE', '公式驱动字段补全', '全为 ', '升级到截面路径重试',
                  '请补正则']:
        print('  probe %-30s -> %d' % (probe, s2.count(probe)))
    print('PATCH OK')


if __name__ == '__main__':
    main()
