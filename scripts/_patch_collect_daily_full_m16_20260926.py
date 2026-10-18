# -*- coding: utf-8 -*-
"""
M16 补丁: collect_daily_full.py 两处 P4 债务 C 残留清零 (字节级, 保持 CRLF)
=========================================================================
目标文件: strategies/final/collect_daily_full.py (CRLF, 405 行)
- 残留①: 末段「完整性检查」仍用 all_days 判定 → 308 个 2027 未来日被误报「仍缺」;
- 残留②: last_update.txt 写入 all_days[-1] = 20271231 (未来日期);
- 附: 债务 C 注释块补一行修复状态说明。

本脚本只做三处**精确字节替换**, 每处要求唯一命中; 完成后校验:
  · CRLF 计数 = 原 CRLF + 新增行数 (旧行被替换为多行时);
  · 裸 LF 仍为 0。
用法: C:/Python314/python.exe scripts/_patch_collect_daily_full_m16_20260926.py
"""
import os
import shutil

SRC = 'e:/quant/strategies/final/collect_daily_full.py'
TMP = SRC + '.tmp_m16'
BAK = SRC + '.bak_m16_20260926'


def L(text):
    """把 LF 片段转成 CRLF 片段 (文件为 CRLF)。"""
    return text.replace('\n', '\r\n')


PATCHES = [
    # ---- ① 完整性检查口径 ----
    (L("""    # 完整性检查
    for key, (fp, cols, name) in FILES.items():
        miss = len(all_days) - len(get_existing_dates(fp))
        if miss > 0:
            print(f'  ⚠️ {name} 仍缺 {miss} 天')
"""),
     L("""    # 完整性检查
    # M16 (2026-09-26): 口径修正 —— 只对「已发生」交易日判定缺失; 「尚未发生」单列披露、
    #   不计缺失。旧版用 all_days 判定, 把 2027 的 308 个未来日全报成「仍缺 N 天」,
    #   健康的库看起来永远不完整 (P4 债务 C 的残留读数, 本次清零)。
    for key, (fp, cols, name) in FILES.items():
        have = get_existing_dates(fp)
        miss = [d for d in work_days if d not in have]
        if miss:
            print(f'  ⚠️ {name} 仍缺 {len(miss)} 天 (仅计已发生; 最大缺失 {miss[-1]})')
        else:
            print(f'  ✅ {name} 已发生交易日无缺失 ({len(work_days)} 天; '
                  f'{len(future_days)} 天尚未发生未采集)')
""")),
    # ---- ② last_update.txt 写入口径 ----
    (L("""    # 为每日增量更新保存最后采集的日期
    with open(os.path.join(RAW, 'last_update.txt'), 'w') as f:
        f.write(f'last_trade_date={all_days[-1]}\\n')
        f.write(f'update_time={time.strftime("%Y-%m-%d %H:%M")}\\n')

    print(f'最后交易日: {all_days[-1]}')
"""),
     L("""    # 为每日增量更新保存最后采集的日期
    # M16 (2026-09-26): 原写 all_days[-1] = 日历末端(未来), 下游若读会以为未来已有数据;
    #   改为「最后一个已发生交易日」, 并显式披露尚未发生边界 (P4 债务 C 残留清零)。
    with open(os.path.join(RAW, 'last_update.txt'), 'w') as f:
        f.write(f'last_trade_date={work_days[-1]}\\n')
        if future_days:
            f.write(f'not_yet_available_from={future_days[0]}\\n')
        f.write(f'update_time={time.strftime("%Y-%m-%d %H:%M")}\\n')

    print(f'最后已发生交易日: {work_days[-1]} (日历末端 {all_days[-1]} 尚未发生, 未采集)')
""")),
    # ---- ③ 债务 C 注释状态 ----
    (L("""#      · last_update.txt 的 last_trade_date 会被写成**未来日期**。
"""),
     L("""#      · last_update.txt 的 last_trade_date 会被写成**未来日期**。
#    [M16, 2026-09-26: 第②③条在末段的残留 (完整性检查/last_update 写入仍用 all_days)
#     已一并清零 —— 修复见下方 [4] 完成段。]
""")),
]


def main():
    data = open(SRC, 'rb').read().decode('utf-8')
    crlf_before = data.count('\r\n')
    bare_before = data.count('\n') - crlf_before
    assert bare_before == 0, f'前置条件不满足: 存在 {bare_before} 个裸 LF'
    added_crlf = 0
    for i, (old, new) in enumerate(PATCHES, 1):
        n = data.count(old)
        assert n == 1, f'补丁 {i}: 命中 {n} 次 (要求恰好 1 次), 拒绝继续'
        data = data.replace(old, new, 1)
        added_crlf += new.count('\r\n') - old.count('\r\n')
        print(f'  补丁 {i}: OK (+{new.count(chr(13)+chr(10)) - old.count(chr(13)+chr(10))} CRLF 行)')

    crlf_after = data.count('\r\n')
    bare_after = data.count('\n') - crlf_after
    assert crlf_after == crlf_before + added_crlf, 'CRLF 计数不符'
    assert bare_after == 0, f'新增了 {bare_after} 个裸 LF, 拒绝写入'

    shutil.copy2(SRC, BAK)
    with open(TMP, 'wb') as f:
        f.write(data.encode('utf-8'))
    os.replace(TMP, SRC)
    print(f'  CRLF: {crlf_before} → {crlf_after} | 裸 LF: {bare_after} | 备份: {BAK}')
    print('完成。')


if __name__ == '__main__':
    main()
