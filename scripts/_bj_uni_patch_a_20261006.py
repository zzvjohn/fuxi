# -*- coding: utf-8 -*-
"""BJ universe 方案 A 受控 patch: s5_joint_filter.py 加 FUXI_S5_BJ_UNIVERSE 开关

5 处改动:
  P1 A_func  : 函数区插入 _s5_bj_universe / _bj_listing_date_map / _s5_bj_cut_pre_listing
  P2 A_mark  : 口径快照区加 _bj = _s5_bj_universe()
  P3 A_switch: _switches 披露加 "bj_universe"
  P4 A_guard : fixed_shares 分流处 fail-closed guard
  P5 A_price : price 面板接入 listed_only 裁剪

纪律: 备份 -> 逐处断言 count==1 -> 替换 -> 写回 -> 四项对账(bytes/CRLF/裸LF/rrn) + sha
"""
import hashlib
import shutil
from datetime import datetime

SRC = r"E:\quant\research\factor_alchemy\s5_joint_filter.py"
BAK = r"E:\quant\reports\fuxi_v1_bj_universe_20261006\pre_bjcut\s5_joint_filter.py.bak"

raw = open(SRC, "r", encoding="utf-8", newline="").read()
pre = {"bytes": len(raw.encode("utf-8")),
       "crlf": raw.count("\r\n"),
       "bare_lf": raw.count("\n") - raw.count("\r\n"),
       "rrn": raw.count("\r\r\n"),
       "sha": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
print("[before]", pre)

# ── 备份 ──
import os
os.makedirs(os.path.dirname(BAK), exist_ok=True)
shutil.copy2(SRC, BAK)
print("[bak]", BAK)

# ── P1 函数区插入 ──
A_FUNC = ("_AXIS1_CALL_RE = re.compile(r'\\.\\s*\\w+\\s*\\([^()]*\\baxis\\s*=\\s*1\\b')"
          "\n\n\ndef _s5_holdings_mark() -> str:")

NEW_FUNCS = '''_AXIS1_CALL_RE = re.compile(r'\\.\\s*\\w+\\s*\\([^()]*\\baxis\\s*=\\s*1\\b')


# ═══════════════════════════════════════════════════════════════════════════
# BJ universe 批 (P-20261006-006, 2026-10-06): FUXI_S5_BJ_UNIVERSE 时点口径
# ---------------------------------------------------------------------------
# 背景 (M17 §3-§4 已裁决): 920xxx.BJ 的 list_date 是北交所上市日 (如 920914.BJ
#   = 2022-11-11), 而行情行自 2021-01-04 起即存在 (新三板挂牌期); 缺失格 =
#   挂牌期「无成交日」——市场微观结构事实, 非数据洞; 开仓后遇缺失格即
#   held_price_observations_missing 阻断。
#   legacy      : 不裁 (历史行为; 920xxx.BJ 挂牌期行情行照常进入建仓/持有路径)
#   listed_only : 对 .BJ 裁掉 trade_date < list_date 的行 —— 池内每日成员 =
#                 当日已上市者 (上市前不可建仓/不可持有); 无 list_date 的 .BJ
#                 代码 fail-safe 不裁 (保持原路径) 并由计数披露。
#   上市日通道 (§6-A 审计, reports/fuxi_v1_bj_universe_20261006/listing_date_audit.json):
#   delist_dates.csv(351) + data/raw/bj_listing_date_overlay.csv(2) = 353/353 覆盖;
#   与第三方仲裁源 (eastmoney f26) 逐只一致 (351/351, 0 mismatch)。
#   本开关只接 proxy 账本路径; 非 proxy 时 listed_only 显式拒绝 (见 backtest_factor)。
# ═══════════════════════════════════════════════════════════════════════════

def _s5_bj_universe() -> str:
    mode = (os.environ.get('FUXI_S5_BJ_UNIVERSE', 'legacy') or 'legacy').strip().lower()
    if mode not in ('legacy', 'listed_only'):
        raise ValueError('FUXI_S5_BJ_UNIVERSE must be legacy or listed_only')
    return mode


_BJ_LISTING_DATE_CACHE = {}


def _bj_listing_date_map():
    """{ts_code: pd.Timestamp} —— .BJ 上市日 (delist_dates.csv + 审计补录 overlay)。

    只读本地封存文件; 无 list_date 的 .BJ 代码不进入映射 (调用方 fail-safe 不裁)。
    """
    if _BJ_LISTING_DATE_CACHE:
        return _BJ_LISTING_DATE_CACHE
    import csv as _csv_mod
    out = {}
    for _p in (RAW_DIR / 'delist_dates.csv', RAW_DIR / 'bj_listing_date_overlay.csv'):
        if not _p.exists():
            continue
        try:
            with open(_p, 'r', encoding='utf-8-sig', newline='') as _fh:
                for _row in _csv_mod.DictReader(_fh):
                    _code = (_row.get('ts_code') or '').strip()
                    _ld = (_row.get('list_date') or '').strip()
                    if (not _code.endswith('.BJ')) or len(_ld) != 8 or not _ld.isdigit():
                        continue
                    try:
                        out.setdefault(_code, pd.Timestamp(_ld))
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            continue
    _BJ_LISTING_DATE_CACHE.update(out)
    return _BJ_LISTING_DATE_CACHE


def _s5_bj_cut_pre_listing(frame):
    """裁掉 .BJ 股票在 list_date 之前的行情行 (BJ universe listed_only 档)。

    只做减法: 输出行 ⊆ 输入行; 非 .BJ 行逐位不变。无 list_date 的 .BJ 代码
    fail-safe 不裁 (保持 legacy 路径)。
    """
    if frame is None or len(frame) == 0 or 'ts_code' not in frame.columns:
        return frame
    _lds = _bj_listing_date_map()
    if not _lds:
        print('    [S5] BJ-universe listed_only: no listing dates loaded; nothing cut')
        return frame
    _bj = frame['ts_code'].astype(str).str.endswith('.BJ')
    if not _bj.any():
        return frame
    _cut = frame['ts_code'].map(_lds)
    _drop = _bj & _cut.notna() & (frame['trade_date'] < _cut)
    _n = int(_drop.sum())
    if _n:
        _codes = int(frame.loc[_drop, 'ts_code'].nunique())
        print('    [S5] BJ-universe listed_only: cut %d pre-listing rows across %d BJ codes'
              % (_n, _codes))
    return frame[~_drop].copy()


def _s5_holdings_mark() -> str:'''

reps = []

reps.append(("P1_func", A_FUNC, NEW_FUNCS))

# ── P2 快照 ──
A_MARK = "        _mark = _s5_holdings_mark()\n        # M6b:"
N_MARK = ("        _mark = _s5_holdings_mark()\n"
          "        _bj = _s5_bj_universe()\n"
          "        # M6b:")
reps.append(("P2_mark", A_MARK, N_MARK))

# ── P3 披露 ──
A_SW = '                     "holdings_mark": _mark}'
N_SW = '                     "holdings_mark": _mark, "bj_universe": _bj}'
reps.append(("P3_switch", A_SW, N_SW))

# ── P4 guard ──
A_GD = ('        if _s5_ledger_mode() != "proxy":\n'
        '            return self._backtest_factor_fixed_shares(factor_df)')
N_GD = ('        if _s5_ledger_mode() != "proxy":\n'
        '            # BJ 批 (P-20261006-006): listed_only 只接到 proxy 路径; 非 proxy\n'
        '            # 若静默不裁会让开关"写着有效实际不生效" ⇒ fail-closed 显式拒绝。\n'
        '            if _s5_bj_universe() == \'listed_only\':\n'
        '                raise ValueError(\n'
        '                    \'FUXI_S5_BJ_UNIVERSE=listed_only currently covers only \'\n'
        '                    \'the proxy ledger path; refusing to run the fixed-shares \'\n'
        '                    \'path with a silently ineffective switch\')\n'
        '            return self._backtest_factor_fixed_shares(factor_df)')
reps.append(("P4_guard", A_GD, N_GD))

# ── P5 裁剪接入 ──
A_PR = ('        price = self._price_df[self._price_df["ts_code"].isin(self._universe)][\n'
        '            [c for c in _pcols if c in self._price_df.columns]\n'
        '        ].copy()')
N_PR = ('        price = self._price_df[self._price_df["ts_code"].isin(self._universe)][\n'
        '            [c for c in _pcols if c in self._price_df.columns]\n'
        '        ].copy()\n'
        '\n'
        '        # ── BJ universe 时点裁剪 (P-20261006-006; legacy=默认零回归) ──\n'
        '        # listed_only 档: .BJ 上市前行情行从建仓/持有路径裁除 (merged 为\n'
        '        # inner join ⇒ 上市前不可进入 entry_data, 持仓标记的预期格同步排除)。\n'
        '        if _bj != \'legacy\':\n'
        '            price = _s5_bj_cut_pre_listing(price)')
reps.append(("P5_price", A_PR, N_PR))

# ── 逐处替换（CRLF 化 + 断言） ──
cur = raw
for name, a, n in reps:
    a_c = a.replace("\n", "\r\n")
    n_c = n.replace("\n", "\r\n")
    assert "\r" not in n.replace("\r\n", ""), f"{name}: new text has bare CR"
    cnt = cur.count(a_c)
    assert cnt == 1, f"{name}: anchor count = {cnt} (expect 1)"
    cur = cur.replace(a_c, n_c)
    print(f"[patch] {name} ok (+{len(n_c) - len(a_c)} chars)")

# ── 写回 ──
out_bytes = cur.encode("utf-8")
with open(SRC, "wb") as f:
    f.write(out_bytes)

post = {"bytes": len(out_bytes),
        "crlf": cur.count("\r\n"),
        "bare_lf": cur.count("\n") - cur.count("\r\n"),
        "rrn": cur.count("\r\r\n"),
        "sha": hashlib.sha256(out_bytes).hexdigest()}
print("[after ]", post)

# ── 对账 ──
exp_delta = sum(len(n.replace("\n", "\r\n")) - len(a.replace("\n", "\r\n"))
                for _, a, n in reps)
print(f"[check ] d_bytes={post['bytes'] - pre['bytes']} (expect {exp_delta})")
print(f"[check ] d_crlf={post['crlf'] - pre['crlf']} | bare_lf {pre['bare_lf']}->{post['bare_lf']} | rrn {pre['rrn']}->{post['rrn']}")
assert post["bare_lf"] == 0 and post["rrn"] == 0, "line ending regression"
print("[ok] patch applied; sha", post["sha"][:16])
