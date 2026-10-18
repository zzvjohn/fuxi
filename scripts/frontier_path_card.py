# -*- coding: utf-8 -*-
"""
P-20260910-006 前沿→路径卡片 (Frontier Path Card)

把 Phase 2.2 的日报前沿条目 (信息层) 升级为路径卡片 (方向层):
每条前沿强制标注三问 — 可数据化? / 可因子化? / 需要什么新数据?

产出: data/frontier_path_cards.json (增量累积, 按 title 去重)
消费: ralph_loop._phase_retrieve 按范式缺口召回相关卡片注入 llm 上下文
      (开关 FRONTIER_CARD_INJECT, 默认 "1"=注入; "0"=关闭 → 退化为现有日报流程)

原始报告条目格式 (2026-09-10 起): "**X1. 标题（来源）**" 独立成行 + 内容行。
本模块只读日报, 不修改日报与 RAG 脚本。

CLI:
    python scripts/frontier_path_card.py --date 2026-09-10            # LLM 标注 (批量 1 次调用)
    python scripts/frontier_path_card.py --date 2026-09-10 --no-llm   # 纯启发式 (离线/无配额)
    python scripts/frontier_path_card.py --stats                      # 打印卡片库统计
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "daily"
CARDS_PATH = DATA_DIR / "frontier_path_cards.json"
REPORT_TEXT_PATH = DATA_DIR / "frontier_path_cards_report.md"  # 日报 2.x 节尾部追加用

# ── 数据表名 → 关键词 (供"可数据化"启发式判定) ───────────────────────
KNOWN_TABLES = {
    "daily_prices": ["价格", "量价", "成交量", "换手", "涨跌", "收盘", "开盘", "振幅", "动量", "反转", "波动"],
    "daily_basic": ["估值", "市盈", "市净", "PE", "PB", "股息", "流通市值", "总市值", "换手率"],
    "moneyflow_daily": ["资金流", "主力", "大单", "小单", "净流入", "净流出", "机构"],
    "moneyflow_hsgt": ["北向", "陆股通", "沪股通", "深股通", "沪深港通"],
    "hk_hold": ["北向持股", "外资持股", "陆股通持股"],
    "report_rc": ["研报", "评级", "目标价", "盈利预测", "分析师", "一致预期", "上调", "下调"],
    "forecast": ["业绩预告", "预增", "预减", "盈利预告"],
    "stk_factor": ["技术因子", "MACD", "KDJ", "RSI", "BOLL"],
    "margin_detail": ["两融", "融资", "融券", "担保", "杠杆"],
    "top_list": ["龙虎榜", "涨停", "席位"],
    "top_inst": ["机构席位", "席位", "游资", "敢死队"],
    "weekly_prices": ["周频", "周线", "周度"],
    "index_daily": ["指数", "大盘", "沪深300", "中证500", "中证1000"],
    "industry": ["行业", "板块", "申万", "轮动"],
}

# ── 算子族 → 关键词 (供"可因子化"启发式判定) ────────────────────────
OPERATOR_FAMILIES = {
    "rolling.rank": ["排名", "分位", "相对强弱", "横截面", "排序"],
    "pct_change": ["变化率", "增速", "动量", "反转", "涨跌幅"],
    "rolling.mean/std": ["均值", "波动", "标准差", "离散", "稳定性"],
    "rolling.corr": ["相关性", "背离", "同步", "耦合"],
    "ewm": ["平滑", "衰减", "加权"],
    "rolling.max/min": ["通道", "位置", "突破", "新高", "新低"],
    "rolling.sum": ["累计", "聚合", "总"],
    "shift.rank": ["滞后", "领先", "前期"],
    "state_conditional": ["条件", "状态下", "拥挤时", "高换手时", "分层", "门控"],
}

# ── 范式关键词 (召回匹配用) ─────────────────────────────────────────
PARADIGM_KEYWORDS = {
    "流动性×微观结构": ["流动性", "冲击", "非流动性", "Amihud", "微观结构", "冲击成本"],
    "资金流": ["资金流", "主力", "净流入", "大单", "小单", "机构流向"],
    "动量反转": ["动量", "反转", "超跌", "修复", "趋势反转"],
    "行为金融": ["行为", "情绪", "羊群", "过度反应", "处置效应", "锚定"],
    "尾部风险": ["尾部", "极端", "峰谷", "回撤", "崩盘", "黑天鹅"],
    "情绪×日内": ["情绪", "日内", "开盘", "收盘", "尾盘", "隔夜", "跳空"],
    "截面交互": ["截面", "交互", "配对", "同伴", "配对交易", "行业相对"],
    "趋势": ["趋势", "均线", "通道", "新高", "突破", "MA"],
    "下行保护": ["下行", "保护", "防御", "低波", "稳健"],
    "波动率适应": ["波动率", "波动", "体制", "regime", "状态切换"],
    "市场宽度": ["宽度", "涨跌家数", "涨停", "红盘率", "扩散"],
    "结构突变": ["突变", "断裂", "结构变化", "转折点", "changepoint"],
    "筹码分布": ["筹码", "成本", "持仓", "套牢", "获利盘", "分布"],
    "事件驱动": ["事件", "公告", "财报", "分红", "并购", "中报", "季报"],
    "行业轮动": ["行业", "轮动", "板块", "申万", "风格切换"],
    "北向资金": ["北向", "陆股通", "外资", "沪股通", "深股通"],
    "两融信号": ["两融", "融资", "融券", "杠杆", "担保比例"],
    "大宗交易": ["大宗", "折价", "溢价", "协议转让"],
    "高频微观结构": ["高频", "分钟", "tick", "盘口", "买卖盘", "order book"],
    "跨资产联动": ["跨资产", "债券", "期货", "商品", "汇率", "联动"],
    "市场情绪综合": ["市场情绪", "综合情绪", "恐慌", "贪婪", "情绪指数"],
}


# ══════════════════════════════════════════════════════════════════
# 1. 日报前沿条目解析
# ══════════════════════════════════════════════════════════════════

_ITEM_HEAD_RE = re.compile(r"^\*\*([A-Z]\d+)\.\s*(.+?)\*\*\s*$")
_BODY_LINE_RE = re.compile(r"^\*\*([A-Z]\d+)\.\s*(.+?)\*\*[:：]\s*(.*)$")


def parse_frontier_items(report_text: str) -> list:
    """从日报 '## 2. 本周量化前沿' 解析前沿条目。

    兼容两种格式:
      (a) 独立行: '**A1. 标题（来源）**' + 后续内容行   (2026-09-10 起)
      (b) 单行:   '**A1. 标题（来源）**: 内容'
    返回 [{code, title, source, body, section}, ...]
    """
    # 截取第 2 章
    m = re.search(r"^##\s*2\.[^\n]*\n(.*?)(?=^##\s*3\.|\Z)", report_text, re.S | re.M)
    section_text = m.group(1) if m else ""
    if not section_text:
        return []

    items = []
    cur_section = ""
    lines = section_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        sub = re.match(r"^###\s*(2\.\d+\s*.+)$", line)
        if sub:
            cur_section = sub.group(1).strip()
            i += 1
            continue

        # 格式 (b): 单行
        mb = _BODY_LINE_RE.match(line)
        if mb:
            items.append(_mk_item(mb.group(1), mb.group(2), mb.group(3), cur_section))
            i += 1
            continue

        # 格式 (a): 独立行 + 后续内容行
        mh = _ITEM_HEAD_RE.match(line)
        if mh:
            body_parts = []
            j = i + 1
            while j < len(lines):
                nl = lines[j].rstrip()
                if _ITEM_HEAD_RE.match(nl) or _BODY_LINE_RE.match(nl) or re.match(r"^###", nl):
                    break
                if nl.strip():
                    body_parts.append(nl.strip())
                j += 1
            items.append(_mk_item(mh.group(1), mh.group(2), " ".join(body_parts), cur_section))
            i = j
            continue
        i += 1
    return items


def _mk_item(code, title, body, section):
    src = ""
    ms = re.search(r"[（(]([^（）()]+)[）)]\s*$", title)
    if ms:
        src = ms.group(1).strip()
        title_clean = title[:ms.start()].strip()
    else:
        title_clean = title.strip()
    return {
        "code": code,
        "title": title_clean,
        "source": src,
        "body": body.strip(),
        "section": section,
    }


# ══════════════════════════════════════════════════════════════════
# 2. 启发式标注 (离线兜底 / LLM 失败时降级)
# ══════════════════════════════════════════════════════════════════

def _hits(text: str, kws) -> list:
    t = text.lower()
    return [k for k in kws if k.lower() in t]


def heuristic_annotate(item: dict) -> dict:
    text = f"{item['title']} {item['body']}"
    tables = [_t for _t, kws in KNOWN_TABLES.items() if _hits(text, kws)]
    ops = [_o for _o, kws in OPERATOR_FAMILIES.items() if _hits(text, kws)]
    paradigms = [_p for _p, kws in PARADIGM_KEYWORDS.items() if _hits(text, kws)]

    data_able = len(tables) > 0
    factorizable = len(ops) > 0 or len(paradigms) > 0
    # 新数据需求: 命中"接口/另类数据/未接入"等词且无本地表命中
    new_data_kw = ["另类数据", "卫星", "舆情", "新闻", "ESG", "供应链", "订单", "接口", "分钟", "tick", "期权", "期货"]
    new_hits = _hits(text, new_data_kw)
    cols = new_hits if (new_hits and not data_able) else []
    return {
        "data_able": {"yes": data_able, "tables": tables[:4]},
        "factorizable": {"yes": factorizable, "operator_families": ops[:4]},
        "new_data_needed": {
            "none": len(cols) == 0,
            "columns": cols,
            "interfaces": [],
        },
        "paradigm_hint": paradigms[0] if paradigms else "",
        "paradigm_hints": paradigms[:3],
        "annotator": "heuristic",
    }


# ══════════════════════════════════════════════════════════════════
# 3. LLM 批量标注 (每日 1 次调用)
# ══════════════════════════════════════════════════════════════════

_SYS_PROMPT = (
    "你是量化因子研究员。给定多条量化前沿条目, 为每条回答路径卡片三问, "
    "只输出 JSON 数组, 不要任何解释或 markdown 围栏。"
)


def _build_annotate_prompt(items: list) -> str:
    lines = [
        "对下列每条前沿条目输出一个 JSON 对象, 数组顺序与输入一致。字段:",
        '{"i": 序号, "data_able": true/false, "tables": ["本地表名"], '
        '"factorizable": true/false, "operators": ["算子族"], '
        '"new_data_none": true/false, "new_data": ["需要的新数据列/接口"], "paradigm": "最贴近的范式名"}',
        "",
        "本地可用表: " + ", ".join(sorted(KNOWN_TABLES.keys())),
        "可选范式: " + ", ".join(sorted(PARADIGM_KEYWORDS.keys())),
        "",
        "条目:",
    ]
    for i, it in enumerate(items):
        lines.append(f"[{i}] ({it['code']}) {it['title']} :: {it['body'][:220]}")
    return "\n".join(lines)


def llm_annotate(items: list) -> list:
    """批量 LLM 标注; 失败抛异常由调用方降级。"""
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "factor_alchemy"))
    from llm_client import get_llm_client  # type: ignore
    client = get_llm_client()
    resp = client.chat_with_system(
        _SYS_PROMPT, _build_annotate_prompt(items), temperature=0.0,
    )
    if isinstance(resp, dict):
        resp = resp.get("content", "") or resp.get("text", "")
    txt = str(resp).strip()
    # 剥离 markdown 围栏
    txt = re.sub(r"^```(?:json)?", "", txt).strip()
    txt = re.sub(r"```$", "", txt).strip()
    start, end = txt.find("["), txt.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("LLM 未返回 JSON 数组")
    arr = json.loads(txt[start:end + 1])
    out = []
    for i, obj in enumerate(arr):
        base = heuristic_annotate(items[i]) if i < len(items) else heuristic_annotate(items[0])
        base["data_able"] = {
            "yes": bool(obj.get("data_able", base["data_able"]["yes"])),
            "tables": list(obj.get("tables", base["data_able"]["tables"]))[:4],
        }
        base["factorizable"] = {
            "yes": bool(obj.get("factorizable", base["factorizable"]["yes"])),
            "operator_families": list(obj.get("operators", base["factorizable"]["operator_families"]))[:4],
        }
        base["new_data_needed"] = {
            "none": bool(obj.get("new_data_none", True)),
            "columns": list(obj.get("new_data", []))[:5],
            "interfaces": [],
        }
        base["paradigm_hint"] = obj.get("paradigm", base["paradigm_hint"])
        base["paradigm_hints"] = [base["paradigm_hint"]] if base["paradigm_hint"] else []
        base["annotator"] = "llm"
        out.append(base)
    if len(out) != len(items):
        raise ValueError(f"LLM 返回条数不匹配 {len(out)} != {len(items)}")
    return out


def annotate_items(items: list, use_llm: bool = True) -> tuple:
    """返回 (annotations, annotator_used)"""
    if use_llm and items:
        try:
            return llm_annotate(items), "llm"
        except Exception as e:
            print(f"  [FrontierCard] LLM 标注失败 → 降级启发式: {type(e).__name__}: {e}")
    return [heuristic_annotate(it) for it in items], "heuristic"


# ══════════════════════════════════════════════════════════════════
# 4. 卡片构建 / 落盘 (增量累积)
# ══════════════════════════════════════════════════════════════════

def build_cards(items: list, annotations: list, date: str, annotator: str) -> list:
    cards = []
    for it, an in zip(items, annotations):
        digest = hashlib.md5(
            (it["code"] + "|" + it["title"]).encode("utf-8")
        ).hexdigest()[:8]
        cards.append({
            "card_id": f"FPC-{date.replace('-', '')}-{it['code']}-{digest}",
            "date": date,
            "code": it["code"],
            "title": it["title"],
            "source": it["source"],
            "section": it["section"],
            "body_summary": it["body"][:400],
            "path_three_questions": {
                "data_able": an["data_able"],
                "factorizable": an["factorizable"],
                "new_data_needed": an["new_data_needed"],
            },
            "paradigm_hint": an.get("paradigm_hint", ""),
            "paradigm_hints": an.get("paradigm_hints", []),
            "annotator": annotator,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
    return cards


def load_cards() -> dict:
    if CARDS_PATH.exists():
        try:
            return json.loads(CARDS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": "1.0.0", "cards": []}


def save_cards(store: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def merge_cards(store: dict, new_cards: list) -> int:
    """按 card_id 去重合并; 返回新增数。"""
    existing = {c.get("card_id") for c in store.get("cards", [])}
    added = 0
    for c in new_cards:
        if c["card_id"] not in existing:
            store.setdefault("cards", []).append(c)
            existing.add(c["card_id"])
            added += 1
    store["version"] = store.get("version", "1.0.0")
    store["last_updated"] = datetime.now().isoformat(timespec="seconds")
    return added


# ══════════════════════════════════════════════════════════════════
# 5. 召回 (ralph_loop 注入用)
# ══════════════════════════════════════════════════════════════════

def recall_cards(paradigm_gaps=None, k: int = 3) -> list:
    """按范式缺口召回相关路径卡片 (纯读, 无副作用)。"""
    store = load_cards()
    cards = store.get("cards", [])
    if not cards:
        return []
    gaps = [g for g in (paradigm_gaps or []) if g]
    if not gaps:
        return cards[-k:]
    scored = []
    for c in cards:
        hints = set(c.get("paradigm_hints") or ([c.get("paradigm_hint")] if c.get("paradigm_hint") else []))
        text = f"{c.get('title','')} {c.get('body_summary','')}"
        score = 2 * len(hints & set(gaps))
        for g in gaps:
            score += sum(1 for kw in PARADIGM_KEYWORDS.get(g, []) if kw.lower() in text.lower())
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def format_for_prompt(cards: list) -> str:
    if not cards:
        return ""
    lines = ["## 前沿路径卡片 (知识注入 → 候选方向, 仅作灵感, 不改变探索选择)"]
    for c in cards:
        q = c.get("path_three_questions", {})
        da = q.get("data_able", {})
        fa = q.get("factorizable", {})
        nd = q.get("new_data_needed", {})
        lines.append(
            f"- [{c.get('paradigm_hint') or '未映射'}] {c.get('title','')[:60]} | "
            f"可数据化={'Y' if da.get('yes') else 'N'}({','.join(da.get('tables', [])[:2])}) "
            f"可因子化={'Y' if fa.get('yes') else 'N'}({','.join(fa.get('operator_families', [])[:2])}) "
            f"新数据={'无' if nd.get('none') else ','.join(nd.get('columns', [])[:2])}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 6. 统计 (日报 2.x 节尾部 / Ralph 汇报)
# ══════════════════════════════════════════════════════════════════

def stats() -> dict:
    store = load_cards()
    cards = store.get("cards", [])
    total = len(cards)
    data_able = sum(1 for c in cards if c.get("path_three_questions", {}).get("data_able", {}).get("yes"))
    factorizable = sum(1 for c in cards if c.get("path_three_questions", {}).get("factorizable", {}).get("yes"))
    from collections import Counter
    paradigm_counter = Counter(c.get("paradigm_hint") for c in cards if c.get("paradigm_hint"))
    return {
        "total": total,
        "data_able": data_able,
        "factorizable": factorizable,
        "both": sum(1 for c in cards
                    if c.get("path_three_questions", {}).get("data_able", {}).get("yes")
                    and c.get("path_three_questions", {}).get("factorizable", {}).get("yes")),
        "top_paradigms": paradigm_counter.most_common(5),
        "last_updated": store.get("last_updated", ""),
        "uncovered_hits": sum(
            1 for c in cards
            if c.get("paradigm_hint") in
            ("高频微观结构", "北向资金", "跨资产联动")
        ),
    }


def stats_line() -> str:
    s = stats()
    tops = " / ".join(f"{k}×{v}" for k, v in s["top_paradigms"][:3]) or "无"
    return (f"路径卡片: 累计 {s['total']} 张 | 可数据化 {s['data_able']} / "
            f"可因子化 {s['factorizable']} / 双可行 {s['both']} | "
            f"未覆盖范式命中 {s['uncovered_hits']} | Top 范式: {tops}")


# ── 日报 2 章尾部落款 (幂等: 有则替换, 无则插入) ───────────────────
_STATS_BEGIN = "<!-- FRONTIER_CARD_STATS:BEGIN -->"
_STATS_END = "<!-- FRONTIER_CARD_STATS:END -->"


def append_stats_to_report(report_path: Path, date: str) -> bool:
    """在第 2 章末尾 (下一个 '## ' 前) 写入/更新卡片统计块。"""
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    block = (f"{_STATS_BEGIN}\n"
             f"**路径卡片统计（P-20260910-006，{date}）**: {stats_line()}\n"
             f"{_STATS_END}")
    # 已有 → 替换
    if _STATS_BEGIN in text and _STATS_END in text:
        new_text = re.sub(
            re.escape(_STATS_BEGIN) + r".*?" + re.escape(_STATS_END),
            block, text, flags=re.S,
        )
        if new_text != text:
            report_path.write_text(new_text, encoding="utf-8")
        return True
    # 插入到第 2 章末尾 (即第 3 个 '## ' 之前)
    m = re.search(r"^##\s*2\.", text, re.M)
    if not m:
        return False
    nm = re.search(r"^##\s*(?!2\.)", text[m.end():], re.M)
    insert_at = m.end() + nm.start() if nm else len(text)
    prefix = text[:insert_at].rstrip("\n")
    suffix = text[insert_at:]
    report_path.write_text(f"{prefix}\n\n{block}\n{suffix}", encoding="utf-8")
    return True


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="前沿→路径卡片生成/统计")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--report", default="", help="日报路径 (默认 reports/daily/{date}_strategy_research.md)")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM, 纯启发式标注")
    ap.add_argument("--stats", action="store_true", help="只打印统计")
    ap.add_argument("--recall", default="", help="按逗号分隔的范式缺口召回并打印")
    ap.add_argument("--no-report", action="store_true", help="不向日报 2 章尾写统计块")
    args = ap.parse_args()

    if args.stats:
        print(stats_line())
        return 0
    if args.recall:
        gaps = [g.strip() for g in args.recall.split(",") if g.strip()]
        cards = recall_cards(gaps, k=5)
        print(format_for_prompt(cards) or "(无命中卡片)")
        return 0

    rp = Path(args.report) if args.report else (REPORT_DIR / f"{args.date}_strategy_research.md")
    if not rp.exists():
        print(f"[FrontierCard] 日报不存在: {rp} → 跳过")
        return 1
    text = rp.read_text(encoding="utf-8")
    items = parse_frontier_items(text)
    if not items:
        print("[FrontierCard] 未解析到前沿条目 → 跳过")
        return 1

    anns, annotator = annotate_items(items, use_llm=not args.no_llm)
    cards = build_cards(items, anns, args.date, annotator)
    store = load_cards()
    added = merge_cards(store, cards)
    save_cards(store)
    print(f"[FrontierCard] 解析 {len(items)} 条 / 标注({annotator}) / 新增卡片 {added} "
          f"/ 累计 {len(store['cards'])}")
    print("  " + stats_line())
    if not args.no_report:
        if append_stats_to_report(rp, args.date):
            print(f"  [FrontierCard] 统计块已写入 {rp.name} 第 2 章尾")
    return 0


if __name__ == "__main__":
    sys.exit(main())
