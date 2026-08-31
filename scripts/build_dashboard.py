# -*- coding: utf-8 -*-
"""
伏羲 v0.5 工作台看板生成器
======================
从本地数据文件聚合生成自包含 HTML 看板（无外部 CDN 依赖，离线可用）。

数据源:
  1. data/stage1_optimization_proposals.json   — Stage 1 优化提案
  2. reports/daily/YYYY-MM-DD_strategy_research.md — Stage 1 日报 (取最新)
  3. data/experience_memory.json               — Memory v3.1 三层反馈
  4. data/mab_scheduler_state.json             — MAB 调度器
  5. data/library_orthogonality_state.json     — Red Sea / 因子库正交性
  6. data/trajectory_log.json                  — 进化轨迹 (EvoTraj)
  7. data/decay_monitor_log.json               — 衰减监控
  8. data/jq_feedback_summary.json             — JQ 反馈闭环 (D+)
  9. data/passed_factor_pool.csv               — 因子库 (JQ 候选)
 10. data/ralph_loop_result_*.json             — Ralph Loop 最近运行

输出: reports/dashboard.html
用法: python scripts/build_dashboard.py [--out reports/dashboard.html]
"""
import json
import glob
import os
import re
import sys
from datetime import datetime

try:
    import pandas as pd
    HAS_PD = True
except Exception:
    HAS_PD = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports", "daily")
OUT_DEFAULT = os.path.join(ROOT, "reports", "dashboard.html")

# ---------------------------------------------------------------- 数据加载

def load_json(rel, default=None):
    p = os.path.join(DATA, rel)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def latest_file(pattern):
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    return files[-1] if files else None


def latest_report():
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "*_strategy_research.md")))
    return files[-1] if files else None


# ---------------------------------------------------------------- Markdown 解析

def md_sections(text):
    """按 '## ' 二级标题切块, 返回 {标题: 内容}"""
    parts = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            parts[cur] = []
        elif cur is not None:
            parts[cur].append(line)
    return {k: "\n".join(v) for k, v in parts.items()}


def md_sub_sections(text):
    """按 '### ' 三级标题切块"""
    parts, cur = {}, None
    for line in text.splitlines():
        if line.startswith("### "):
            cur = line[4:].strip()
            parts[cur] = []
        elif cur is not None:
            parts[cur].append(line)
    return {k: "\n".join(v) for k, v in parts.items()}


def md_tables(text):
    """解析 markdown 表格 → list[dict] (多表合并, 含表头行检测)"""
    rows = []
    cur_header = None
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            cur_header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cur_header is None:
            cur_header = cells
        elif set(cells) <= {"---", "---:", ":---", "------"} or all(
                re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        else:
            d = {}
            for i, c in enumerate(cells):
                if i < len(cur_header):
                    d[cur_header[i]] = c
            rows.append(d)
    return rows


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------- 各模块提取

def get_proposals():
    props = load_json("stage1_optimization_proposals.json", [])
    if isinstance(props, dict):
        props = props.get("proposals", [])
    return props or []


def get_system_snapshot(report_text):
    """从日报 '1. 伏羲系统快照' 提取 KPI 表 + 未覆盖范式 + 摘要"""
    out = {"kpi": {}, "uncovered": [], "summary": "", "engine_mix": "—"}
    secs = md_sections(report_text or "")
    snap = secs.get("1. 伏羲系统快照", "")
    subs = md_sub_sections(snap)
    for title, content in subs.items():
        for row in md_tables(content):
            key = row.get("指标", "")
            val = row.get("数值", "")
            chg = row.get("变化", "")
            if key:
                out["kpi"][key] = {"v": val, "c": chg}
    # 未覆盖范式
    m = re.search(r"未覆盖范式\s*[（(]?\d+[)）]?\s*[:\s]*(.+)", snap, re.S)
    if not m:
        m = re.search(r"未覆盖范式[^\n]*\n((?:[-–•]\s*.+\n?)+)", snap)
    if m:
        out["uncovered"] = [re.sub(r"^[-–•\s]*\d*\.?\s*", "", x).strip()
                            for x in re.split(r"\n", m.group(1)) if x.strip()][:8]
    # 摘要
    m2 = re.search(r"系统快照摘要[^\n]*\n\s*>\s*(.+)", snap)
    if m2:
        out["summary"] = m2.group(1).strip()
    # 引擎分配
    m3 = re.search(r"引擎分配\s*\|\s*([^|\n]+)\|", snap)
    if m3:
        out["engine_mix"] = m3.group(1).strip()
    return out


def get_frontier(report_text):
    """从日报 '2. 本周量化前沿' 提取 4 维度条目"""
    secs = md_sections(report_text or "")
    front = secs.get("2. 本周量化前沿", "") or secs.get("2. 量化前沿", "")
    dims = []
    for title, content in md_sub_sections(front).items():
        items = []
        # 条目: "1. **标题**: 内容" 或 "1. 标题 — 内容"
        for m in re.finditer(r"(\d+)\.\s*\*\*(.+?)\*\*[:\s]*(.*?)(?=\n\s*\d+\.\s|\n\s*$|$)", content, re.S):
            items.append({"title": m.group(2).strip(), "body": m.group(3).strip().replace("\n", " ")})
        if not items:
            for m in re.finditer(r"(\d+)\.\s*([^:：\n]+)[:：\s]+(.*?)(?=\n\s*\d+\.\s|$)", content, re.S):
                items.append({"title": m.group(2).strip(), "body": m.group(3).strip().replace("\n", " ")})
        if items:
            dims.append({"dim": title, "items": items[:4]})
    return dims


def get_risks_and_actions(report_text):
    """从日报 '4. 风险预警' + '5. 下一步行动建议' 提取"""
    secs = md_sections(report_text or "")
    risks, actions = [], []
    for title, content in md_sub_sections(secs.get("4. 风险预警", "")).items():
        body = content.strip()
        if body:
            risks.append({"title": title, "body": body})
    for line in (secs.get("5. 下一步行动建议", "") or "").splitlines():
        m = re.match(r"\d+\.\s*\*?\*?([^*\n]+?)\*?\*?\s*[::]\s*(.+)", line.strip())
        if m:
            actions.append({"title": m.group(1).strip(), "body": m.group(2).strip()})
        elif re.match(r"\d+\.\s*\*", line.strip()):
            actions.append({"title": line.strip().lstrip("0123456789. *"), "body": ""})
    return risks, actions


def get_memory_state():
    mem = load_json("experience_memory.json", {})
    attempts = mem.get("attempts", [])
    forb = mem.get("forbidden_directions", [])
    warn = mem.get("warning_directions", [])
    motifs = mem.get("motif_rules", [])
    if isinstance(motifs, list):
        forbid_n = len([m for m in motifs if m.get("type") == "forbid"])
        prefer_n = len([m for m in motifs if m.get("type") == "prefer"])
    else:
        forbid_n = prefer_n = 0
    succ = mem.get("success_templates", [])
    n_pass = sum(1 for a in attempts if a.get("passed") or a.get("outcome") in ("PASS", "passed"))
    return {
        "attempts": len(attempts),
        "pass_rate": (n_pass / len(attempts)) if attempts else 0,
        "forbidden": len(forb),
        "warnings": len(warn),
        "motif_forbid": forbid_n,
        "motif_prefer": prefer_n,
        "success_templates": len(succ),
    }


def get_mab_state():
    mab = load_json("mab_scheduler_state.json", {})
    dirs = mab.get("directions", [])
    if isinstance(dirs, dict):
        dirs = list(dirs.values())
    active = [d for d in dirs if d.get("status") == "active"]
    cooling = [d for d in dirs if d.get("status") == "cooling"]
    top = sorted(dirs, key=lambda d: -float(d.get("expected_reward") or 0))[:6]
    # 按范式聚合 top (去 model_type 后缀)
    seen, top_agg = set(), []
    for d in top:
        pid = d.get("paradigm") or d.get("direction_id", "?")
        if pid in seen:
            continue
        seen.add(pid)
        top_agg.append({"paradigm": pid, "reward": round(float(d.get("expected_reward") or 0), 3),
                        "pulls": d.get("pulls", 0)})
    return {
        "total_pulls": mab.get("total_pulls", 0),
        "active": len(active),
        "cooling": len(cooling),
        "top": top_agg[:5],
    }


def get_library_state():
    orth = load_json("library_orthogonality_state.json", {})
    stats = orth.get("stats", {})
    by_para = stats.get("by_paradigm", {})
    by_status = stats.get("by_status", {})
    return {
        "updated": (orth.get("updated_at") or "")[:16],
        "n_factors": stats.get("total_factors", 0),
        "n_clusters": stats.get("n_clusters", 0),
        "by_paradigm": by_para,
        "by_status": by_status,
        "forbidden_regions": len(orth.get("forbidden_regions", [])),
    }


def get_red_sea(report_text):
    """从日报提取 Red Sea 行; 缺省 fallback"""
    secs = md_sections(report_text or "")
    snap = secs.get("1. 伏羲系统快照", "")
    for row in md_tables(snap):
        if "Red Sea" in str(row.get("指标", "")):
            return str(row.get("数值", ""))
    return "—"


def get_market_crowding():
    """P-20260815-003: 读取市场级拥挤度监控读数 (纯监控)"""
    c = load_json("market_crowding.json", None)
    if not c or not isinstance(c, dict):
        return None
    return c


def render_market_crowding(c):
    """市场拥挤监控卡片行 (纯读数, 带提醒色, 无干预说明)"""
    if not c:
        return ('<div class="sys-item"><span class="sys-k">市场拥挤度</span>'
                '<span class="sys-v">未生成 · 运行 scripts/market_crowding_monitor.py</span></div>')
    conc = c.get("top5_concentration_20d")
    lsr = c.get("ls_vol_ratio_20d")
    alerts = set(c.get("alerts", []))
    conc_s = f"{conc:.1%}" if conc is not None else "—"
    lsr_s = f"{lsr:.2f}" if lsr is not None else "—"
    warn = " ⚠️" if alerts else ""
    return (f'<div class="sys-item"><span class="sys-k">市场拥挤度{warn}</span>'
            f'<span class="sys-v">{c.get("date", "—")} · 集中度 {conc_s} · 多空波动比 {lsr_s}<br>'
            f'<small>阈值: 集中度&gt;50% / 波动比&gt;1.5 (仅提醒, 不干预管线)</small></span></div>')


def get_trajectory_state():
    traj = load_json("trajectory_log.json", {})
    trajs = traj.get("trajectories", [])
    jq = [t for t in trajs if t.get("jq_validated")]
    dplus = [t for t in trajs if t.get("d_plus_applied")]
    outcomes = {}
    for t in trajs:
        o = t.get("final_outcome") or "?"
        outcomes[o] = outcomes.get(o, 0) + 1
    return {
        "total": traj.get("total_trajectories", len(trajs)),
        "jq_validated": len(jq),
        "d_plus": len(dplus),
        "outcomes": outcomes,
    }


def get_decay_state():
    decay = load_json("decay_monitor_log.json", {})
    scans = decay.get("scans", [])
    if not scans:
        return {"structural": 0, "linear": 0, "alerts": []}
    last = scans[-1]
    lin = last.get("linear_alerts", [])
    stru = last.get("structural_alerts", [])
    alerts = []
    for a in (stru + lin):
        alerts.append({"factor": a.get("factor_name", "?"),
                       "type": a.get("alert_type", "?"),
                       "severity": a.get("severity", "?"),
                       "detail": a.get("detail", "")})
    return {"structural": len(stru), "linear": len(lin), "alerts": alerts[:6]}


def get_jq_feedback():
    jqf = load_json("jq_feedback_summary.json", {})
    return {
        "generated": (jqf.get("generated_at") or "")[:10],
        "breeds": jqf.get("breeds_tested", 0),
        "findings": jqf.get("key_findings", []) or [],
        "mab_update": jqf.get("mab_update", {}),
    }


def get_ralph_latest():
    f = latest_file("ralph_loop_result_*.json")
    if not f:
        return None
    with open(f, encoding="utf-8") as fh:
        r = json.load(fh)
    ev = (r.get("stages_completed") or {}).get("evaluate") or {}
    return {
        "ts": os.path.basename(f).replace("ralph_loop_result_", "").replace(".json", ""),
        "source": r.get("source", "?"),
        "elapsed": round(r.get("elapsed_seconds", 0)),
        "candidates": r.get("total_candidates", 0),
        "s1": ev.get("stage1_passed", "?"),
        "s5": ev.get("stage5_passed", "?"),
        "eligible": r.get("n_jq_candidates", 0),
        "jq_names": r.get("jq_candidates", []),
    }


def get_jq_verified_names():
    """已做过 JQ 回测的因子名集合 = experience_memory 中 jq_verified/jq_return 非空 + jq_codegen_queue 中已有 jq 结果的键"""
    names = set()
    mem = load_json("experience_memory.json", {})
    for a in mem.get("attempts", []) or []:
        if a.get("jq_verified") or a.get("jq_return") is not None:
            n = a.get("factor_name")
            if n:
                names.add(str(n))
    queue = load_json("jq_codegen_queue.json", {})
    for k, v in (queue or {}).items():
        if v.get("jq_result") or str(v.get("status", "")).startswith("jq_run_done"):
            names.add(str(k))
    return names


def get_jq_pending(pool_df):
    """JQ 待验证因子: status∈{candidate, s5_passed} 且尚未 JQ 回测过（排除已 JQ 验证/已出结果的因子与无指标的历史遗留条目）"""
    items = []
    excluded = 0
    if HAS_PD and pool_df is not None and len(pool_df):
        cand = pool_df[pool_df["status"].astype(str).str.lower().isin(["candidate", "s5_passed"])]
        verified = get_jq_verified_names()
        for _, row in cand.iterrows():
            name = str(row.get("name", "?"))
            # 已 JQ 回测过的 → 不再展示为"待验证"
            if name in verified:
                excluded += 1
                continue
            # 无任何本地检验指标的历史遗留脏条目（如 capital_efficiency_proxy）→ 不展示
            icir = row.get("icir")
            ic_mean = row.get("ic_mean")
            if (icir is None or (isinstance(icir, float) and icir != icir)) and \
               (ic_mean is None or (isinstance(ic_mean, float) and ic_mean != ic_mean)):
                excluded += 1
                continue
            items.append({
                "name": name,
                "label": str(row.get("label", "")) or name,
                "paradigm": str(row.get("category", "")) or "—",
                "formula": str(row.get("formula", "")) or "—",
                "meaning": str(row.get("logic", "")) or str(row.get("hypothesis", "")) or "—",
                "ic": ic_mean,
                "icir": icir,
                "plus_pct": row.get("+ic_pct"),
                "direction": str(row.get("direction", "")),
                "date": str(row.get("date", "")) or "",
            })
        # 新通过候选排前（date 降序，缺失日期沉底）
        items.sort(key=lambda x: x["date"], reverse=True)
    return items, excluded


# ---------------------------------------------------------------- 渲染

RISK_BADGE = {"low": ("低风险", "#16a34a", "#dcfce7"),
              "medium": ("中风险", "#d97706", "#fef3c7"),
              "high": ("高风险", "#dc2626", "#fee2e2")}

CAT_COLORS = {
    "因子扩展": "#4f46e5", "管线增强": "#0891b2", "新数据源": "#7c3aed",
    "策略优化": "#0d9488", "参数调整": "#ca8a04", "LLM公式生成": "#db2777",
    "FactorForge调参": "#ea580c",
}


def fmt_num(x, nd=3):
    try:
        v = float(x)
        if v != v:  # NaN
            return "—"
        return f"{v:.{nd}f}"
    except Exception:
        return "—"


def render_kpi(label, value, sub="", tone="normal"):
    tone_cls = f"kpi-{tone}"
    sub_html = f'<div class="kpi-sub">{esc(sub)}</div>' if sub else ""
    return (f'<div class="kpi {tone_cls}"><div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{esc(str(value))}</div>{sub_html}</div>')


def render_proposals(props):
    if not props:
        return '<div class="empty">暂无提案</div>'
    # 只展示尚未审核/落地的提案（status 非空 = 已 implemented/tested/monitor 落地处理，不再展示）
    pending = [p for p in props if not (p.get("status") or "").strip()]
    # 新提案排前（proposal_id 倒序）
    pending.sort(key=lambda p: str(p.get("proposal_id", "")), reverse=True)
    cards = []
    for p in pending[:8]:
        cat = p.get("category", "—")
        gen = p.get("generator_hint", "auto")
        risk = p.get("risk_level", "medium")
        rb = RISK_BADGE.get(risk, RISK_BADGE["medium"])
        cat_c = CAT_COLORS.get(cat, "#64748b")
        cards.append(f"""
        <div class="prop-card">
          <div class="prop-head">
            <span class="prop-id">{esc(p.get('proposal_id','?'))}</span>
            <span class="badge" style="background:{cat_c}22;color:{cat_c};border:1px solid {cat_c}55">{esc(cat)}</span>
            <span class="badge" style="background:#eef2ff;color:#4f46e5;border:1px solid #c7d2fe">🧬 {esc(gen)}</span>
            <span class="badge" style="background:{rb[2]};color:{rb[1]};border:1px solid {rb[1]}44">{rb[0]}</span>
          </div>
          <div class="prop-title">{esc(p.get('title',''))}</div>
          <div class="prop-line"><b>触发:</b> {esc(str(p.get('trigger',''))[:180])}</div>
          <div class="prop-line"><b>做法:</b> {esc(str(p.get('proposed_change',''))[:180])}</div>
          <div class="prop-line"><b>预期:</b> {esc(str(p.get('expected_impact',''))[:150])}</div>
        </div>""")
    return "".join(cards)


def render_frontier(dims):
    if not dims:
        return '<div class="empty">日报暂无前沿扫描内容</div>'
    blocks = []
    dim_emoji = {"A": "📈", "B": "🤖", "C": "⚠️", "D": "🛰️"}
    for d in dims:
        tag = ""
        m = re.search(r"维度\s*([A-D])", d["dim"])
        if m:
            tag = dim_emoji.get(m.group(1), "📌")
        items_html = []
        for it in d["items"]:
            items_html.append(
                f'<div class="front-item"><span class="front-dot"></span>'
                f'<div><div class="front-title">{esc(it["title"])}</div>'
                f'<div class="front-body">{esc(it["body"][:260])}</div></div></div>')
        blocks.append(f'<div class="front-card"><div class="front-dim">{tag} {esc(d["dim"])}</div>'
                      f'{"".join(items_html)}</div>')
    return "".join(blocks)


def render_system(mem, mab, lib, traj, decay, ralph, red_sea, snapshot, market_crowding=None):
    # 范式覆盖条形图
    paras = sorted(lib["by_paradigm"].items(), key=lambda x: -x[1])[:12]
    maxn = max([n for _, n in paras] or [1])
    bars = []
    for name, n in paras:
        w = max(4, int(n / maxn * 100))
        bars.append(f'<div class="bar-row"><span class="bar-name">{esc(name)}</span>'
                    f'<div class="bar-track"><div class="bar-fill" style="width:{w}%"></div></div>'
                    f'<span class="bar-num">{n}</span></div>')
    eng = snapshot.get("engine_mix", "—")
    ralph_html = ""
    if ralph:
        ralph_html = f"""
        <div class="sys-item"><span class="sys-k">Ralph Loop</span>
          <span class="sys-v">最近 {esc(ralph['ts'])} · {esc(ralph['source'])} · {ralph['elapsed']}s<br>
          <small>候选 {ralph['candidates']} · S1通过 {ralph['s1']} · S5通过 {ralph['s5']} · JQ候选 {ralph['eligible']}</small></span></div>"""
    decay_html = ""
    if decay["alerts"]:
        alerts_html = "".join(
            f'<div class="decay-alert {esc(a["severity"])}">{esc(a["factor"])}: {esc(a["detail"][:60])}</div>'
            for a in decay["alerts"][:4])
        decay_html = f'<div class="sys-item"><span class="sys-k">衰减监控</span><span class="sys-v">{alerts_html}</span></div>'
    return f"""
    <div class="sys-grid">
      <div class="sys-item"><span class="sys-k">Memory v3.1</span>
        <span class="sys-v">{mem['attempts']} 尝试 · 通过率 {mem['pass_rate']:.1%}<br>
        <small>硬禁止 {mem['forbidden']} · 软警告 {mem['warnings']} · motif F/P {mem['motif_forbid']}/{mem['motif_prefer']} · 模板 {mem['success_templates']}</small></span></div>
      <div class="sys-item"><span class="sys-k">MAB 调度</span>
        <span class="sys-v">{mab['total_pulls']} 拉取 · 活跃 {mab['active']} · 冷却 {mab['cooling']}<br>
        <small>引擎: {esc(eng)}</small></span></div>
      <div class="sys-item"><span class="sys-k">因子库</span>
        <span class="sys-v">{lib['n_factors']} 因子 · {lib['n_clusters']} 族 · 禁区 {lib['forbidden_regions']}<br>
        <small>状态: reserve {lib['by_status'].get('reserve',0)} / candidate {lib['by_status'].get('candidate',0)}</small></span></div>
      <div class="sys-item"><span class="sys-k">Red Sea</span>
        <span class="sys-v">{esc(red_sea)}</span></div>
      {ralph_html}
      {render_market_crowding(market_crowding)}
      <div class="sys-item"><span class="sys-k">进化轨迹</span>
        <span class="sys-v">{traj['total']} 轨迹 · JQ验证 {traj['jq_validated']} · D+蒸馏 {traj['d_plus']}</span></div>
      {decay_html}
    </div>
    <div class="para-chart">
      <div class="para-title">范式因子分布 Top-12</div>
      {''.join(bars)}
    </div>"""


def render_jq_pending(items, jq_feedback, excluded=0):
    note = ('<div class="jq-note">⚠️ 校验层级: 本区因子 = 通过 Stage 2 日频本地检验 (ICIR≥0.3 且 +IC%≥55%) 且 <b>尚未在聚宽回测</b> 的候选。'
            '已做过 JQ 回测的因子（含失败/边际/通过）与无指标的历史遗留条目已自动过滤' +
            (f'（本次过滤 {excluded} 条）' if excluded else '') +
            '。本地 IC/ICIR 仅作入场券、不做排序 —— 按 JQ 铁律, 聚宽实盘验证才是唯一真相源 (local→JQ gap 1.4~4.6x)。'
            'S1-S5 全管线 eligible 名单另见 data/ralph_loop_result_*.json 的 jq_candidates。</div>')
    if not items:
        return note + '<div class="empty">暂无 JQ 待验证候选因子</div>'
    rows = []
    for it in items:
        direction = "多" if it["direction"] == "long" else ("空" if it["direction"] == "short" else "—")
        dir_cls = "dir-long" if it["direction"] == "long" else "dir-short"
        rows.append(f"""
        <div class="jq-card">
          <div class="jq-head">
            <span class="jq-name">{esc(it['label'])}</span>
            <span class="badge" style="background:#eef2ff;color:#4f46e5;border:1px solid #c7d2fe">{esc(it['paradigm'])}</span>
            <span class="badge {dir_cls}">{direction}</span>
            <span class="jq-metrics">
              <span class="metric">IC {fmt_num(it['ic'])}</span>
              <span class="metric">ICIR {fmt_num(it['icir'])}</span>
              <span class="metric">+IC% {fmt_num(it['plus_pct'], 1)}</span>
            </span>
          </div>
          <div class="jq-meaning">📖 {esc(it['meaning'][:220])}</div>
          <div class="jq-formula"><code>{esc(it['formula'][:260])}</code></div>
        </div>""")
    fb = ""
    if jq_feedback["findings"]:
        fb = ('<div class="jq-fb"><b>最近 JQ 反馈 (D+)</b> · ' + esc(jq_feedback["generated"]) +
              f' · {esc(str(jq_feedback["breeds"]))} 个 breed 已验证<br>' +
              "".join(f'<div class="fb-line">• {esc(x)}</div>' for x in jq_feedback["findings"][:4]) +
              '</div>')
    return note + "".join(rows) + fb


def render_problems(risks, actions, decay, snapshot):
    items = []
    # 从日报风险提取
    for r in risks:
        body = r["body"]
        lines = [re.sub(r"^[-–•]\s*", "", x).strip()
                 for x in body.splitlines() if x.strip() and not x.strip().startswith("#")]
        text = " ".join(lines)[:220]
        if text:
            items.append({"sev": "high", "title": r["title"], "body": text})
    # 衰减告警
    for a in decay["alerts"]:
        sev = "high" if a["severity"] == "danger" else "mid"
        items.append({"sev": sev, "title": f"衰减告警: {a['factor']} ({a['type']})",
                      "body": a["detail"]})
    # 引擎失衡
    eng = snapshot.get("engine_mix", "")
    if "100% gp_breed" in eng or ("llm" in eng and "0%" in eng):
        items.append({"sev": "high", "title": "三引擎失衡",
                      "body": f"引擎分配 {eng} — LLM/Forge 引擎闲置，需手动干预打破冷启动循环"})
    # 行动建议
    for a in actions[:5]:
        items.append({"sev": "mid", "title": f"行动: {a['title']}", "body": a["body"][:180]})
    if not items:
        return '<div class="empty">暂无记录</div>'
    sev_dot = {"high": "#dc2626", "mid": "#d97706", "low": "#16a34a"}
    return "".join(
        f'<div class="prob-item"><span class="prob-dot" style="background:{sev_dot.get(i["sev"],"#64748b")}"></span>'
        f'<div><div class="prob-title">{esc(i["title"])}</div>'
        f'<div class="prob-body">{esc(i["body"])}</div></div></div>'
        for i in items[:14])


CSS = """
:root {
  --bg: #f4f6fb; --card: #ffffff; --ink: #1e293b; --ink2: #475569; --ink3: #94a3b8;
  --line: #e2e8f0; --brand: #4f46e5; --brand2: #6366f1; --ok: #16a34a; --warn: #d97706; --bad: #dc2626;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  background: var(--bg); color: var(--ink); font-size: 14px; line-height: 1.55; }
.wrap { max-width: 1360px; margin: 0 auto; padding: 20px 22px 40px; }
header.dash-head { display:flex; justify-content:space-between; align-items:flex-end;
  padding: 18px 4px 14px; border-bottom: 2px solid var(--line); margin-bottom: 16px; flex-wrap: wrap; gap: 8px;}
.dash-title { font-size: 24px; font-weight: 700; letter-spacing: .5px; }
.dash-title .sub { font-size: 13px; color: var(--ink2); font-weight: 500; margin-left: 10px; }
.dash-meta { color: var(--ink3); font-size: 12.5px; text-align: right; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 14px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.kpi-label { font-size: 12px; color: var(--ink2); margin-bottom: 3px; }
.kpi-value { font-size: 21px; font-weight: 700; }
.kpi-sub { font-size: 11.5px; color: var(--ink3); margin-top: 2px; }
.kpi-warn .kpi-value { color: var(--warn); } .kpi-bad .kpi-value { color: var(--bad); }
.kpi-ok .kpi-value { color: var(--ok); }
.grid { display: grid; grid-template-columns: 1fr 380px; gap: 14px; align-items: start; }
@media (max-width: 1080px) { .grid { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(15,23,42,.05); }
.card h2 { font-size: 16px; margin-bottom: 12px; display:flex; align-items:center; gap:8px; }
.card h2 .tag { font-size: 11px; color: var(--ink3); font-weight: 400; }
.empty { color: var(--ink3); padding: 14px 0; text-align:center; }
/* 提案 */
.prop-card { border: 1px solid var(--line); border-left: 3px solid var(--brand2);
  border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; background: #fafbff; }
.prop-head { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-bottom: 5px; }
.prop-id { font-family: Consolas, monospace; font-size: 11.5px; color: var(--brand); font-weight: 600; }
.badge { font-size: 11px; padding: 1.5px 8px; border-radius: 20px; white-space: nowrap; }
.prop-title { font-weight: 650; margin-bottom: 4px; }
.prop-line { font-size: 12.3px; color: var(--ink2); margin-top: 2px; }
.prop-line b { color: var(--ink); }
/* 前沿 */
.front-card { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
.front-dim { font-weight: 650; margin-bottom: 6px; color: #312e81; }
.front-item { display: flex; gap: 8px; padding: 4px 0; font-size: 12.6px; }
.front-dot { flex: 0 0 7px; width: 7px; height: 7px; border-radius: 50%; background: var(--brand2); margin-top: 6px; }
.front-title { font-weight: 600; }
.front-body { color: var(--ink2); }
/* 系统 */
.sys-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.sys-item { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; background: #fbfcfe; }
.sys-k { display:block; font-size: 11px; color: var(--ink3); margin-bottom: 2px; }
.sys-v { font-size: 12.6px; font-weight: 600; }
.sys-v small { font-weight: 400; color: var(--ink2); }
.decay-alert { font-size: 11.5px; padding: 2px 6px; border-radius: 6px; margin-top: 3px; font-weight: 500; }
.decay-alert.danger { background: #fee2e2; color: var(--bad); }
.decay-alert.warn { background: #fef3c7; color: var(--warn); }
.para-chart { margin-top: 12px; }
.para-title { font-size: 12px; color: var(--ink2); font-weight: 600; margin-bottom: 8px; }
.bar-row { display: grid; grid-template-columns: 108px 1fr 26px; gap: 8px; align-items: center; margin-bottom: 4px; }
.bar-name { font-size: 11.5px; color: var(--ink2); text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { background: #eef1f7; border-radius: 6px; height: 9px; overflow: hidden; }
.bar-fill { background: linear-gradient(90deg, var(--brand), var(--brand2)); height: 100%; border-radius: 6px; }
.bar-num { font-size: 11px; color: var(--ink3); }
/* JQ */
.jq-card { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; background: #fdfdff; }
.jq-head { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-bottom: 6px; }
.jq-name { font-weight: 650; font-size: 13.5px; }
.jq-metrics { margin-left: auto; display: flex; gap: 6px; }
.metric { font-size: 11.5px; background: #f1f5f9; border-radius: 6px; padding: 2px 7px; color: var(--ink2); font-family: Consolas, monospace; }
.dir-long { background: #fee2e2 !important; color: #b91c1c !important; border: 1px solid #fca5a5 !important; }
.dir-short { background: #dcfce7 !important; color: #15803d !important; border: 1px solid #86efac !important; }
.jq-meaning { font-size: 12.6px; color: var(--ink2); margin-bottom: 5px; }
.jq-formula code { font-size: 11.3px; background: #f1f5f9; border-radius: 6px; padding: 4px 8px;
  display: inline-block; color: #334155; word-break: break-all; max-width: 100%; }
.jq-fb { margin-top: 8px; border: 1px dashed #c7d2fe; background: #f5f7ff; border-radius: 10px; padding: 10px 12px; font-size: 12.5px; }
.jq-note { margin-bottom: 10px; border: 1px solid #fde68a; background: #fffbeb; border-radius: 10px; padding: 9px 12px; font-size: 12.3px; color: #92400e; }
.fb-line { color: var(--ink2); margin-top: 3px; }
/* 问题 */
.prob-item { display: flex; gap: 10px; padding: 8px 2px; border-bottom: 1px dashed var(--line); }
.prob-item:last-child { border-bottom: none; }
.prob-dot { flex: 0 0 9px; width: 9px; height: 9px; border-radius: 50%; margin-top: 6px; }
.prob-title { font-weight: 600; font-size: 13px; }
.prob-body { font-size: 12.4px; color: var(--ink2); }
footer.dash-foot { margin-top: 18px; color: var(--ink3); font-size: 12px; text-align: center; }
a { color: var(--brand); text-decoration: none; }
"""


def build_html():
    now = datetime.now()
    report_path = latest_report()
    report_text = ""
    report_name = "—"
    if report_path:
        report_name = os.path.basename(report_path)
        with open(report_path, encoding="utf-8") as f:
            report_text = f.read()

    proposals = get_proposals()
    snapshot = get_system_snapshot(report_text)
    dims = get_frontier(report_text)
    risks, actions = get_risks_and_actions(report_text)
    mem = get_memory_state()
    mab = get_mab_state()
    lib = get_library_state()
    traj = get_trajectory_state()
    decay = get_decay_state()
    jqf = get_jq_feedback()
    ralph = get_ralph_latest()
    red_sea = get_red_sea(report_text)
    market_crowding = get_market_crowding()

    pool = None
    if HAS_PD:
        csv_path = os.path.join(DATA, "passed_factor_pool.csv")
        if os.path.exists(csv_path):
            try:
                pool = pd.read_csv(csv_path)
            except Exception:
                pool = None
    jq_items, jq_excluded = get_jq_pending(pool)

    n_uncovered = len(snapshot["uncovered"])
    red_tone = "ok"
    if "RED" in red_sea or "🔴" in red_sea:
        red_tone = "bad"
    elif "YELLOW" in red_sea or "🟡" in red_sea:
        red_tone = "warn"
    decay_tone = "ok"
    if decay["alerts"]:
        decay_tone = "bad" if any(a["severity"] == "danger" for a in decay["alerts"]) else "warn"

    kpis = "".join([
        render_kpi("因子库", lib["n_factors"], f"{lib['n_clusters']} 族 · 更新 {lib['updated']}"),
        render_kpi("进化轨迹", traj["total"], f"JQ 验证 {traj['jq_validated']} · D+ 蒸馏 {traj['d_plus']}"),
        render_kpi("Memory 尝试", mem["attempts"], f"通过率 {mem['pass_rate']:.1%}"),
        render_kpi("JQ 待验证", len(jq_items), f"候选因子 {lib['by_status'].get('candidate', 0)} · 已过滤 {jq_excluded}"),
        render_kpi("Red Sea", red_sea.split("(")[0].strip() if red_sea != "—" else "—",
                   f"禁区 {lib['forbidden_regions']}", red_tone),
        render_kpi("未覆盖范式", n_uncovered, f"覆盖 {21 - n_uncovered}/21"),
        render_kpi("衰减告警", f"{decay['structural'] + decay['linear']}",
                   f"结构性 {decay['structural']} · 线性 {decay['linear']}", decay_tone),
        render_kpi("待审核提案", len(proposals), f"最新日报 {report_name[:10]}"),
    ])

    pipeline = f"""
    <div class="card">
      <h2>📦 流水线状态</h2>
      <div class="sys-grid">
        <div class="sys-item"><span class="sys-k">Stage 0 数据采集</span><span class="sys-v">工作日 08:00 · Tushare 增量</span></div>
        <div class="sys-item"><span class="sys-k">Stage 1 研究提案</span><span class="sys-v">每日 09:30 · 日报 + 提案</span></div>
        <div class="sys-item"><span class="sys-k">Stage 2 因子实验</span><span class="sys-v">每日 13:00 · S1-S5 验证</span></div>
        <div class="sys-item"><span class="sys-k">Stage 3 自进化</span><span class="sys-v">每日 15:00 · Ralph Loop</span></div>
        <div class="sys-item"><span class="sys-k">RAG 入库</span><span class="sys-v">日报 → 本地 RAG 索引</span></div>
        <div class="sys-item"><span class="sys-k">看板刷新</span><span class="sys-v">流水线末步自动更新</span></div>
      </div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>伏羲 v0.5 工作台 · {now:%Y-%m-%d}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="dash-head">
    <div class="dash-title">📊 伏羲 v0.5 量化研究工作台<span class="sub">Fuxi Quant Pipeline · Memory v3.1 + 三引擎</span></div>
    <div class="dash-meta">数据截至 {esc(report_name) if report_name != '—' else '—'}<br>看板生成: {now:%Y-%m-%d %H:%M:%S}</div>
  </header>

  <div class="kpis">{kpis}</div>

  <div class="grid">
    <div>
      <div class="card">
        <h2>🧠 量化研究前沿 <span class="tag">来自最新 Stage 1 日报 · 4 维度扫描</span></h2>
        {render_frontier(dims)}
      </div>
      <div class="card">
        <h2>🧪 需去 JQ 验证的因子 <span class="tag">仅显示未回测候选 · 已回测因子自动过滤</span></h2>
        {render_jq_pending(jq_items, jqf, jq_excluded)}
      </div>
    </div>
    <div>
      <div class="card">
        <h2>🚀 主要提案 <span class="tag">仅显示未落地提案 · 待用户审核</span></h2>
        {render_proposals(proposals)}
      </div>
      <div class="card">
        <h2>⚙️ 伏羲系统运行状况</h2>
        {render_system(mem, mab, lib, traj, decay, ralph, red_sea, snapshot, market_crowding)}
      </div>
      <div class="card">
        <h2>🔥 急需解决的问题</h2>
        {render_problems(risks, actions, decay, snapshot)}
      </div>
      {pipeline}
    </div>
  </div>

  <footer class="dash-foot">
    由 scripts/build_dashboard.py 自动生成 · 数据源: data/*.json + passed_factor_pool.csv + reports/daily/*.md
    · 每轮伏羲流水线结束后自动刷新
  </footer>
</div>
</body>
</html>"""
    return html


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT_DEFAULT
    html = build_html()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[dashboard] 已生成: {out} ({len(html):,} bytes)")
    # 摘要
    report_path = latest_report()
    print(f"[dashboard] 数据源日报: {os.path.basename(report_path) if report_path else '无'}")
    props = get_proposals()
    print(f"[dashboard] 提案 {len(props)} 条 · 前沿维度 {len(get_frontier((open(report_path, encoding='utf-8').read()) if report_path else ''))} 个")


if __name__ == "__main__":
    main()
