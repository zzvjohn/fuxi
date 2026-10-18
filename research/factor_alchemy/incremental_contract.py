# -*- coding: utf-8 -*-
"""P6 -- 增量对照的冻结设计契约。

为什么需要它
------------
干旱复核的第 4 项把「一个冻结设计、三个组合器」落成了
``scripts/fuxi_model_compare_20260924.py``。它的冻结守卫是**手写的**：重冻结只允许
 plumbing 修复，办法是把 18 个 ``DESIGN_FIELDS`` 逐项比对，非空就抛错。这条守卫有三处
不够（不是错，是不够）：

1. **它只在一个批次脚本里**。全仓 14 份 ``reports/*/protocol.json`` 中，只有
   ``model_compare_20260924`` 带 ``revisions``，其余 13 份没有任何「相对基准改了什么」
   的记账通道。同一个纪律换一个批次就不存在了。
2. **它只有 0 与非 0 两档**。于是「只增删一个特征」这种**合法**的单维增量，和
   「同时改了特征集与 Top80」这种**污染**，在守卫眼里是同一件事：都抛错。真正的增量
   实验因此只能靠人自觉不去碰第二个维度。
3. **它按字段名比，而字段名跨层会重名**。``pool`` 在 ``model_compare`` 里是选股规则
   （字符串），在 ``target_diagnostics`` 里是名字池读数；``status`` 在 R5 里是预注册状态。
   按名字一刀切会把「规则」和「读数」混成一类 —— 这正是 [mem:AE4] 记的那类坑。

它做什么
--------
把「相对基准只允许改一个维度」从**靠自觉**变成**靠契约**，且契约为纯函数、可机检：

* 协议里的每个键被归入 DESIGN / BOOKKEEPING / READING 三类中的**恰好一类**；
  **未登记的键一律按 DESIGN 处理**（闸门只向下锁，从不授予）。
* 增量最多只能碰**一个轴**（axis）。轴是「按构造就会一起动」的设计字段的声明式分组：
  加一个候选特征会同时动 ``features`` / ``dropped_by_coverage`` / ``dropped_by_redundancy`` /
  ``selection_rules``，把它们分成四个维度会把这个**单一决策**误读成四处改动。
* 落到**叶级增量**必须满足：被触碰的**特征个数** ≤ ``max_features_touched``（默认 1）。
  增、删、改（direction/kind/source 任一）都各算触碰一次；改名 = 1 增 + 1 删 = 触碰 2 个
  ⇒ 拒绝（fail-closed，想合法就分两步走，或显式放宽阈值 —— 放宽得写下来）。
* ``features[i].coverage`` 是**读数**不是设计。数据一延伸它就会变；若把它算进设计比较，
  每次补数据都会把「同一次增量」读成一次设计变更 —— 与 [mem:AD5]「补数据会让接近平衡点的
  读数退回」是同一根刺。契约把它单独归到 READING 并报出来。
* 判定：``IDENTICAL`` / ``SINGLE_AXIS`` / ``MULTI_AXIS`` / ``AXIS_ITEM_BATCH``；
  ``accepted`` 只在前两者为真，且 **``accepted`` 对变更集单调不增**：往一个已拒绝的对照里
  再加改动，永远不会把它变成可接受（单向下锁，单测钉死）。

它不做什么
----------
不跑实验、不读任何标签、不翻任何默认、不产生任何裁决。它只对**两份已冻结协议的字典**做
形状判定，并把它看见的东西原样报出来。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    'BOOKKEEPING_KEYS',
    'FEATURE_DESIGN_KEYS',
    'FEATURE_READING_KEYS',
    'MODEL_COMPARE_AXES',
    'MODEL_COMPARE_BASELINE',
    'IncrementSpecError',
    'axes_of',
    'check_increment',
    'classify_key',
    'declared_names',
    'design_fields_from_source',
    'feature_delta',
    'protocol_sha256',
]


class IncrementSpecError(ValueError):
    """轴声明本身有问题。声明了却对不上 ⇒ 抛错，不静默忽略。"""


# --------------------------------------------------------------------------- 分类
# 全局兜底：只有「来源 / 时间 / 摘要 / 审计轨迹」这类**无论如何都不是设计选择**的名字
# 才放进这里。其余一切（含跨层会重名的字段）由调用方的 Spec 显式声明 —— 见 [mem:AE4]。
BOOKKEEPING_KEYS = frozenset({
    # 身份
    'batch', 'purpose', 'supersedes', 'inherited_contract',
    # 时间
    'created_at', 'preregistered_at',
    # 摘要
    'hashes', 'code_hashes', 'data_hashes', 'sha256', 'code_sha256',
    'signal_hashes', 'original_file_hashes', 'prior_audit_sha256',
    'prior_input_hashes', 'prior_actions_sha256', 'prior_protocol_sha256',
    'parent_protocol_sha256',
    # 审计轨迹（本模块自己写进去的记账键也在其中，否则它会让每一次修订都自我拒绝）
    'revisions', 'revision', 'status',
    # 生产版本镜像（记的是「当时生产是什么」，不是设计选择）
    'production_version', 'production_defaults_changed',
})

FEATURE_DESIGN_KEYS = ('name', 'kind', 'direction', 'source')
FEATURE_READING_KEYS = ('coverage',)


def classify_key(key: str, spec: Optional[Mapping[str, Any]] = None) -> str:
    """把一个协议键归入 DESIGN / BOOKKEEPING / READING 之一。

    优先级：Spec 的显式声明 > 全局兜底名字集 > DESIGN（fail-closed）。
    """
    if spec:
        if key in tuple(spec.get('bookkeeping_keys', ())):
            return 'BOOKKEEPING'
        if key in tuple(spec.get('reading_keys', ())):
            return 'READING'
    return 'BOOKKEEPING' if key in BOOKKEEPING_KEYS else 'DESIGN'


# --------------------------------------------------------------------------- 轴
# model_compare_20260924 的轴声明：**唯一**一份把「设计维度」写下来的地方。
# 每个轴里的字段按构造就会一起动，所以「动了一个轴」= 一个决策，不是 N 个改动。
MODEL_COMPARE_AXES: Dict[str, Tuple[str, ...]] = {
    'feature_set': ('features', 'dropped_by_coverage', 'dropped_by_redundancy',
                    'selection_rules', 'directions_note'),
    'combiner': ('arms', 'ridge', 'tree'),
    'label': ('label', 'benchmark'),
    'selection': ('selection', 'top_n'),
    'cost': ('cost_bps', 'primary_cost_bps', 'cost_model'),
    'training': ('training', 'segments'),
    'reference': ('reference_arm',),
}

MODEL_COMPARE_BASELINE = 'reports/model_compare_20260924/protocol.json'


def declared_names(spec: Optional[Mapping[str, Any]]) -> set:
    """Spec 里**显式**写下来的名字（轴成员 ∪ 追加记账键 ∪ 追加读数键）。"""
    spec = spec or {}
    names = set(spec.get('bookkeeping_keys', ())) | set(spec.get('reading_keys', ()))
    for members in dict(spec.get('axes', {})).values():
        names |= set(members)
    return names


def axes_of(spec: Optional[Mapping[str, Any]], design_fields: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
    """把 Spec 的轴声明补齐成「覆盖全部设计字段」的映射。

    * Spec 里声明了但**不存在于设计字段集**的名字 ⇒ ``IncrementSpecError``（fail-closed）。
    * 未被任何轴覆盖的设计字段 ⇒ 各自成为一个单字段轴（最严默认）。
    """
    fields = set(design_fields)
    spec = spec or {}
    declared: Dict[str, Tuple[str, ...]] = {}
    for axis, members in dict(spec.get('axes', {})).items():
        members = tuple(members)
        if not members:
            raise IncrementSpecError('轴 %r 没有成员' % axis)
        unknown = [m for m in members if m not in fields]
        if unknown:
            raise IncrementSpecError(
                '轴 %r 声明了不属于设计字段集的名字: %s' % (axis, unknown))
        declared[axis] = members
    seen: Dict[str, str] = {}
    for axis, members in declared.items():
        for member in members:
            if member in seen:
                raise IncrementSpecError(
                    '字段 %r 被轴 %r 与轴 %r 同时声明' % (member, seen[member], axis))
            seen[member] = axis
    for field in sorted(fields - set(seen)):
        declared[field] = (field,)
    return declared


# --------------------------------------------------------------------------- 特征增量
def feature_delta(baseline: Any, candidate: Any,
                  design_keys: Sequence[str] = FEATURE_DESIGN_KEYS) -> Dict[str, Any]:
    """按**特征名**比较两份 ``features``，把设计改动与读数改动分开。

    返回：
      * ``added`` / ``removed``：名字集合的差
      * ``modified``：名字相同但 ``name/kind/direction/source`` 有变的（含逐字段 before/after）
      * ``field_edits``：(名字, 字段) 形式的扁平清单
      * ``touched``：被触碰的**特征个数** = |added| + |removed| + |modified|
      * ``readings_changed``：仅 ``coverage`` 变了的特征名（**不算设计改动**）
      * ``order_changed``：同样的名字集合但顺序不同（顺序是设计的一部分吗？不是 —— 但要说出来）
    """
    base = {str(f.get('name')): dict(f) for f in (baseline or ())}
    cand = {str(f.get('name')): dict(f) for f in (candidate or ())}
    added = sorted(set(cand) - set(base))
    removed = sorted(set(base) - set(cand))
    modified, field_edits, readings_changed = [], [], []
    for name in sorted(set(base) & set(cand)):
        edits = []
        for key in design_keys:
            before, after = base[name].get(key), cand[name].get(key)
            if before != after:
                edits.append({'field': key, 'before': before, 'after': after})
        if edits:
            modified.append(name)
            field_edits.extend({'name': name, **e} for e in edits)
        if base[name].get('coverage') != cand[name].get('coverage'):
            readings_changed.append(name)
    order_changed = ([str(f.get('name')) for f in (baseline or ())]
                     != [str(f.get('name')) for f in (candidate or ())])
    return {
        'added': added,
        'removed': removed,
        'modified': modified,
        'field_edits': field_edits,
        'touched': len(added) + len(removed) + len(modified),
        'readings_changed': sorted(readings_changed),
        'order_changed': bool(order_changed),
    }


# --------------------------------------------------------------------------- 主判定
def _raw_kind_key(value: Any) -> Any:
    """比较用的规范化：不依赖对象身份，也不受 JSON 键序影响。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)


def _design_view(key: str, value: Any, features_key: str) -> Any:
    """把**读数**从待比较的值里剔掉，只留下设计部分。

    🔴 这是契约里最容易被写漏的一处：``features`` 是一整个列表，若按原样比较，那么
    某个特征的 ``coverage`` 一变，整个 ``features`` 字段就被判成「设计改动」——
    于是每次补数据都会把「同一次增量」读成一次设计变更（[mem:AD5] 的同一根刺）。
    故对特征表按「除读数键之外的全部键」做投影，未登记的键一律保留（fail-closed）。
    """
    if key == features_key and isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, Mapping):
                out.append({k: v for k, v in item.items()
                            if k not in FEATURE_READING_KEYS})
            else:
                out.append(item)
        return out
    return value


def check_increment(baseline: Mapping[str, Any], candidate: Mapping[str, Any], *,
                    spec: Optional[Mapping[str, Any]] = None,
                    baseline_sha256: Optional[str] = None,
                    candidate_sha256: Optional[str] = None,
                    max_features_touched: int = 1,
                    feature_axis: str = 'feature_set',
                    compared_axes: bool = True,
                    allow_new_design_keys: bool = False) -> Dict[str, Any]:
    """对一对已冻结协议做增量形状判定。纯函数，零副作用。

    ``spec`` 形状::

        {
          'axes': {轴名: (字段, ...), ...},          # 可选；未覆盖的字段各自成轴
          'bookkeeping_keys': ('...',),             # 可选；追加的记账键
          'reading_keys': ('...',),                 # 可选；追加的读数键
          'features_key': 'features',               # 可选，默认 'features'
        }

    ``allow_new_design_keys`` 默认 False：**只出现在一侧的设计键一律拒绝**。
    理由不是洁癖，是那是一个真实的夹带口子 —— 一个未登记的键名叫 `cfg` 的字段可以
    在内部装下任意多个维度，而按「名字」看它只动了一处。要让它合法，就把名字写下来
    （进轴、或进 bookkeeping/reading），这一步是审计痕迹，不是形式主义。
    """
    base, cand = dict(baseline or {}), dict(candidate or {})
    design_fields = sorted({k for k in list(base) + list(cand)
                            if classify_key(k, spec) == 'DESIGN'})
    axes = axes_of(spec, design_fields) if compared_axes else {f: (f,) for f in design_fields}
    features_key = str((spec or {}).get('features_key', 'features'))
    declared = declared_names(spec)

    changed_design, changed_bookkeeping, changed_reading = [], [], []
    for key in sorted(set(base) & set(cand)):
        if (_raw_kind_key(_design_view(key, base[key], features_key))
                == _raw_kind_key(_design_view(key, cand[key], features_key))):
            continue
        kind = classify_key(key, spec)
        {'DESIGN': changed_design, 'BOOKKEEPING': changed_bookkeeping,
         'READING': changed_reading}[kind].append(key)

    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))
    undeclared = {'only_in_baseline': [k for k in only_base
                                       if classify_key(k, spec) == 'DESIGN'],
                  'only_in_candidate': [k for k in only_cand
                                        if classify_key(k, spec) == 'DESIGN'],
                  'explicitly_declared': [], 'not_explicitly_declared': []}
    # 键集差异里非 DESIGN 的部分也要报出来，只是不参与设计判定
    added_bookkeeping = [k for k in only_cand if classify_key(k, spec) == 'BOOKKEEPING']
    removed_bookkeeping = [k for k in only_base if classify_key(k, spec) == 'BOOKKEEPING']

    delta = feature_delta(base.get(features_key), cand.get(features_key))

    # 「改动了哪些设计维度」= 两侧都在但不同 ∪ 只出现在一侧的未登记名。
    # 只报前者会让「新增一个键」在报告里消失 —— 被排除的项不得从输出里消失。
    design_moved = sorted(set(changed_design) | set(undeclared['only_in_baseline'])
                          | set(undeclared['only_in_candidate']))
    axes_changed = sorted({axis for axis, members in axes.items()
                           if set(members) & set(design_moved)})

    if delta['readings_changed']:
        changed_reading = sorted(set(changed_reading)
                                 | {'%s[].coverage' % features_key})

    reasons: list = []
    flags: list = []
    if undeclared['only_in_baseline'] or undeclared['only_in_candidate']:
        undeclared['explicitly_declared'] = sorted(
            k for k in (undeclared['only_in_baseline'] + undeclared['only_in_candidate'])
            if k in declared)
        undeclared['not_explicitly_declared'] = sorted(
            k for k in (undeclared['only_in_baseline'] + undeclared['only_in_candidate'])
            if k not in declared)
        flags.append({'code': 'UNDECLARED_KEY', 'detail': {
            'only_in_baseline': undeclared['only_in_baseline'],
            'only_in_candidate': undeclared['only_in_candidate'],
            'explicitly_declared': undeclared['explicitly_declared'],
            'not_explicitly_declared': undeclared['not_explicitly_declared'],
            'note': ('键只出现在一侧 ⇒ 按 DESIGN 计。若它其实只是记账/读数，'
                     '就把名字写进 BOOKKEEPING_KEYS / Spec；若它真的是新设计维度，'
                     '就把它写进某个轴。一个名字背后可以装任意多个维度，'
                     '所以「没写下来」这件事本身就要拒绝 —— 闸门只向下锁。')}})
    smuggle = ([] if allow_new_design_keys
               else list(undeclared.get('not_explicitly_declared') or []))
    if smuggle:
        reasons.append('只出现在一侧且未写下来的设计键: %s' % smuggle)
    if delta['order_changed'] and not (delta['added'] or delta['removed']):
        flags.append({'code': 'FEATURE_ORDER_CHANGED', 'detail': {
            'note': '名字集合相同但顺序不同。顺序不作为设计维度，但必须可见。'}})

    if smuggle:
        verdict = 'UNDECLARED_KEY'
    elif not design_moved:
        verdict = 'IDENTICAL'
    elif len(axes_changed) > 1:
        verdict = 'MULTI_AXIS'
        reasons.append('同时改动了 %d 个轴: %s' % (len(axes_changed), axes_changed))
    elif axes_changed == [feature_axis] and delta['touched'] > max_features_touched:
        verdict = 'AXIS_ITEM_BATCH'
        reasons.append('轴 %r 上一次触碰了 %d 个特征 (上限 %d): +%s -%s ~%s'
                       % (feature_axis, delta['touched'], max_features_touched,
                          delta['added'], delta['removed'], delta['modified']))
    else:
        verdict = 'SINGLE_AXIS'

    accepted = verdict in ('IDENTICAL', 'SINGLE_AXIS')
    return {
        'verdict': verdict,
        'accepted': accepted,
        'refuses': not accepted,
        'is_an_increment': verdict == 'SINGLE_AXIS',
        'allow_new_design_keys': bool(allow_new_design_keys),
        'undeclared_design_keys_that_were_not_written_down': sorted(
            undeclared['not_explicitly_declared']),
        'axes_changed': axes_changed,
        'axis_declaration': {axis: list(members) for axis, members in sorted(axes.items())},
        'changed_design_fields': design_moved,
        'changed_design_fields_present_on_both_sides': changed_design,
        'changed_bookkeeping_fields': changed_bookkeeping,
        'changed_reading_fields': changed_reading,
        'undeclared_keys': undeclared,
        'bookkeeping_keys_added': added_bookkeeping,
        'bookkeeping_keys_removed': removed_bookkeeping,
        'feature_delta': delta,
        'max_features_touched': int(max_features_touched),
        'feature_axis': feature_axis,
        'flags': flags,
        'reasons': reasons,
        'baseline_sha256': baseline_sha256,
        'candidate_sha256': candidate_sha256,
        'note': ('accepted 对变更集单调不增：往一个已拒绝的对照里再加改动，'
                 '永远不会变成可接受。'),
    }


# --------------------------------------------------------------------------- 工具
def protocol_sha256(payload: Any) -> str:
    """对协议做**规范化**摘要：JSON 键序无关，供「上一版是谁」记账用。"""
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def design_fields_from_source(path: Any, name: str = 'DESIGN_FIELDS') -> Tuple[str, ...]:
    """从生产脚本的**源码字面量**读出设计字段清单（不 import，不触发重依赖）。

    这样契约声明的字段集与生产脚本里的那份是**同一个字符串**，而不是靠人抄一遍。
    """
    import ast

    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets:
                return tuple(ast.literal_eval(node.value))
    raise IncrementSpecError('在 %s 里找不到 %s' % (path, name))
