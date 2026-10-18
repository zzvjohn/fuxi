# -*- coding: utf-8 -*-
"""LANDMARK 追加 '2026-09-26' 键（同日多批统一声明）。

纪律（2026-09-25 日志先例 + MEMORY 铁律）：
- 键取 proposal_id 日期：P-20260926-018~022 => '2026-09-26'；
  批次日=09-26 且未跨午夜（22:5x 操作），不属于「孤儿键」禁止场景。
- 纯追加：不动既有 15 键一个字节；新键置于块尾（09-24 之后）。
- version_bump.py 纯 LF：读 bytes -> decode -> 操作 -> encode -> 写 bytes，
  不引入任何 CR；备份同行；写后 ast.parse + 键数/长度断言。

用法: python -X utf8 scripts/_patch_landmark_20260926.py
"""
import ast
import hashlib
import io
import os
import shutil
import sys

SRC = 'e:/quant/scripts/version_bump.py'
BAK = SRC + '.bak_landmark_20260926'
KEY = '2026-09-26'

SEGS = [
    '本日两条主线：早批 v0.7.78~83 完成 A0–A6 学习链与交易标签闭合（机制选题、量子挑战臂隔离、五态反馈、'
    '三臂参考模拟账户）；晚批 v0.7.84~88 按《因子干旱全流程诊断与迭代删改方案》'
    '(reports/fuxi_alpha_generation_overhaul_20260926.md) 实施 P0 速赢首批 M16/M12/M7/M8/M9。',
    '两端结论一致：正式 S5/JQ 新增 0、有效经济奖励 0、R5 未知样本 0、干旱未解除；全窗为研究已暴露窗口，'
    '不得重新包装成样本外。晚批严守 P0 边界（M11/M13 等 P1 项未启动），未降 S5 阈值、未改 Forge 默认、'
    '未扩大生成预算。',
    '本日第一批 v0.7.78（P-20260926-001/002）：机制计划自动选题接入 Ralph（FUXI_SEMANTIC_PLAN=auto 且提供'
    '计划与证据文件才启用、默认仍 off；证据索引校验原始 SHA/协议标识/固定股数账本/15bps/同窗配对）；'
    '量子启发式 Forge 挑战臂以隔离方式试跑——5 种子三臂逐种子唯一求值相等（14/13/14/13/11；每臂 65），'
    'QEA 最佳 legacy 分数胜 Forge 3/5 但 4/5 与同编码零旋转随机臂最佳公式和分数完全相同 '
    '=> 无独立更新优势证据；修复 Forge final_population 上代适应度错配。该试跑未事前封存协议、'
    '未算净值与成本，不是经济收益或 S5 裁决。',
    '本日第二~三批 v0.7.79~80（P-20260926-003~013）：A0 默认停止 S1 IC 直改 MAB、整轮最大 IC 归因及 '
    'EvoTraj/L1 未核实奖惩；逐候选保留来源、决策身份与五态。A1 对 229 计划事件重审（Tushare 税字段反用'
    '修复、10 家发行人公告支持差异化参考价/同日合并/纯送转，229/229 只可作税前毛现金压力输入），'
    '严格三臂有限 NAV 96/320、91/320、基线扩展 101/320，均 data_insufficient。A2 Stage2 接统一 '
    'off/shadow/enforce 前检（默认 off）+ S5 结果数不匹配失败关闭 + 单写锁。A3 限制性历史预告补链 '
    '249/249 日历日、新增 4,869 行、原表 SHA 不变。QEA 独立停止位/算子位、256 码归一、24/24 根算子可达。'
    'A4 代理比较 QEA 对同编码随机 2胜2平1负、无独立旋转优势；机制逐时经济比较 0 轮。A5 影子事件/决策'
    '账本跨进程幂等（单写锁下、可选），四轮探索、有效经济奖励 0。A6 独立冻结与 JQ 逐日对拍门 12 项测试'
    '通过，但无新候选未来成熟样本。',
    '本日第四批 v0.7.81（P-20260926-014/015）：三臂交易标签闭合——合并 24 键中 23 键仅税前毛现金压力；'
    '002822.SZ 重整转增不向旧股东分股（适配器返回 not_holder_distribution）；零碎股上海/北京无唯一'
    '公开解（双情景）；诊断发现 300208/600190/300391 退市与 601989 换股；301678 盘中临停由 unknown '
    '改保守 partial_halt；A4 只读准入 blocked_data_insufficient。',
    '本日第五批 v0.7.82（P-20260926-016）：三臂 8/8 情景参考模拟账户均 320/320 日闭合，24 情景共 '
    '7,680 账户日与独立 Decimal 逐日重算（原始开/收、记录日权利、15bps、税率、换股与模型估值）最大 '
    'NAV 绝对差 <1e-7 元；全窗成本后收益 v3 32.003~35.308%、Top80 18.179~18.291%、v2 16.145~16.424%，'
    '同情景 8/8 顺序 v3>Top80>v2；只可称「声明模型下探索性固定组合排序」，同源内部算术不等于平台'
    '独立对拍。',
    '本日第六批 v0.7.83（P-20260926-017）：v3 收益/PIT 审计——69/69 Top80 证券与顺序复算一致、'
    '公式价字段未借后续行情，真实执行与供应商 PIT 仍待验证；v3 相对 Top80 320 日 +138.25 万元，'
    '但前十好日 +351.32 万超过最终差额、两臂同日平均仅重合 3.99/80 只 => 不能直接归因机制；'
    '封存五新种子与四臂等唯一求值预算 13/12/13/13/12、63 条完整公式档案，QEA 与同编码随机冠军'
    '五轮全同（重合 52/63），未证量子式更新优越。',
    '本日晚批 v0.7.84（M16 数据链三缺口）：①指数 000985/000905/000300 纳入日更流水线'
    '（末端 09-09→09-24，+11/+11/+10 行）；②stk_limit trade_date 归一修复（短横线写法 352,991 行'
    '归零、去重后 0 重复，配套审计 JSON 留档 sha 前后）；③sw_daily 申万行业恢复日更'
    '（04-24→09-24，+46,095 行/105 日）；④collect_daily_full v2.1 缺失判定分流（已发生 1,390 / '
    '尚未发生 308，只对已发生判缺失）。三缺口为 DATE_CEIL 评价链前置（评价窗 +11 交易日）。',
    'v0.7.85（M12 DSR 双默认清理）：deflated_sharpe_ratio 与 MultipleTestingGuard 的 min_probability '
    '函数默认 0.90→0.01（与 config.V06_EXPERIMENTAL 主路径生效值对齐；0.90 系例外回退分支残留默认）；'
    '纯清理、零裁决、零行为变化预期。',
    'v0.7.86（M7 批内 AST 去重接主线）：FUXI_BATCH_AST_DEDUP 默认 enforce（与 generate() 内历史无条件'
    '去重一致）；Ralph 生产主线此前逐轮直调 parse_response、从未经过批内去重 => 同批同骨架候选重复'
    '消耗评估预算的缺口本批闭合；结构重复保留先出现者、命中全部留档（不静默丢弃）；范围仅批内、'
    'reviewer 二轮不跨轮去重；12/12 测试。',
    'v0.7.87（M8 S2/S3/S4 空通过诚实化，annotate-only）：S2 相关矩阵来源（s2_corr_source）、'
    'S3 配对分母（s3_pairs_computed/s3_pairs_missing）、S4 替换评估（s4_replacement_eval_status）'
    '全部显式化——空结果不再等于通过；影子计数（m8_shadow_reject_s2/s3/s4_replacement）只记不拦；'
    '16/16 测试。',
    'v0.7.88（M9 消除假门禁）：FSA share 模式实测不可达（max_count=3 < 阈值 42）显式暴露为 '
    'reachable=False 并降级为观察（不伪造可达性、不翻阈值）；统一质量门三态接线 FUXI_QUALITY_GATE_WIRE '
    '默认 shadow（记账留档、零行为变化；enforce 须显式且被拒项完整进 records——被全拒项不得从输出'
    '消失）；单向下锁（只会减少候选）；17/17 测试。',
    '晚批测试：M7/M8/M9 合计 45/45 + Stage2 回归 4/4；实现证据与备份见对应提案'
    '（P-20260926-018~022）与 reports/fuxi_p0_quickwins_impl_20260926.md。',
]


def build_key_block():
    lines = ["    '%s': '%s'" % (KEY, SEGS[0])]
    for s in SEGS[1:]:
        lines.append("                  '%s'" % s)
    return "\n".join(lines) + ","


def main():
    raw = io.open(SRC, 'rb').read()
    sha_before = hashlib.sha256(raw).hexdigest()
    src = raw.decode('utf-8')

    # 幂等守卫
    if ("'%s'" % KEY) in src:
        print('[landmark] ABORT: 已存在 %s 键，不重复追加' % KEY)
        return 2

    # 定位 LANDMARK 块
    i0 = src.find('LANDMARK = {')
    if i0 < 0:
        print('[landmark] ABORT: 未找到 LANDMARK = {')
        return 2
    j0 = src.find('\n}', i0)
    if j0 < 0:
        print('[landmark] ABORT: 未找到块尾')
        return 2
    if src[j0 - 1] != ',' or src[j0 + 1] != '}':
        print('[landmark] ABORT: 块尾形状不符: %r' % src[j0 - 30:j0 + 5])
        return 2

    # 段内禁用字符检查（英文单引号会破坏字符串）
    for k, s in enumerate(SEGS):
        for bad in ("'", '\\'):
            if bad in s:
                print('[landmark] ABORT: 段 %d 含禁用字符 %r' % (k, bad))
                return 2

    # 备份
    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
    bak_sha = hashlib.sha256(io.open(BAK, 'rb').read()).hexdigest()
    if bak_sha != sha_before:
        print('[landmark] ABORT: 备份 sha 与原文件不一致')
        return 2

    block = build_key_block()
    new_src = src[:j0] + '\n' + block + src[j0:]

    # 语法校验
    ast.parse(new_src)

    # 键数 / 文本长度断言（内存里 exec 同一 LANDMARK 定义）
    ns = {}
    code = new_src[new_src.find('LANDMARK = {'):new_src.find('\n}', new_src.find('LANDMARK = {')) + 2]
    exec(compile(code, '<landmark>', 'exec'), ns)
    lm = ns['LANDMARK']
    assert len(lm) == 16, len(lm)
    assert KEY in lm, 'key missing'
    text_len = len(lm[KEY])
    assert text_len > 2500, text_len

    # 写入（bytes, 保 LF）
    if b'\r' in raw:
        print('[landmark] ABORT: 原文件含 CR（预期纯 LF），需人工确认')
        return 2
    io.open(SRC, 'wb').write(new_src.encode('utf-8'))
    raw2 = io.open(SRC, 'rb').read()
    if b'\r' in raw2:
        print('[landmark] ABORT: 写后含 CR'); return 2
    sha_after = hashlib.sha256(raw2).hexdigest()

    print('[landmark] 备份: %s (sha %s)' % (os.path.basename(BAK), bak_sha[:16]))
    print('[landmark] LANDMARK 15 -> %d 键; %s 文本长度 %d 字符' % (len(lm), KEY, text_len))
    print('[landmark] ast.parse OK; sha %s -> %s' % (sha_before[:16], sha_after[:16]))
    print('[landmark] 新块尾部预览:')
    tail = block.split('\n')[-1]
    print('   ', tail[:100] + ('...' if len(tail) > 100 else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
