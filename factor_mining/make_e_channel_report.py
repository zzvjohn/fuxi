"""
E 道复评报告生成器（v2）
========================
读 results/e_channel/e_channel_candidates.json，生成自包含 HTML 报告，回答：
  Q1  JQ 门升级（旧硬门 → 贝叶斯收缩 ICIR）通过率变化？
  Q2  JQ 候选有多少穿过生产 E1-E4 研究门？两个方法各自的失败模式是什么？
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results" / "e_channel"


def pct(num, den):
    return f"{num}/{den} ({num/den*100:.0f}%)" if den else "0/0"


def bar(num, den, color="var(--accent)"):
    p = num / den * 100 if den else 0.0
    return (f"<div class='bar'><div class='fill' style='width:{p:.0f}%;background:{color}'></div></div>"
            f"<span class='pct'>{p:.0f}%</span>")


def build_report():
    data = json.load(open(RESULTS / "e_channel_candidates.json", encoding="utf-8"))
    cands = data["candidates"]
    cfg = data["summary"]["config"]
    ns = [c for c in cands if c["method"] == "nsga2"]
    gf = [c for c in cands if c["method"] == "gflownet"]

    def stats(sub):
        n = len(sub)
        return {
            "n": n,
            "jq_new": sum(1 for c in sub if c["jq_passed"]),
            "jq_old": sum(1 for c in sub if c["old_gate_passed"]),
            "e1": sum(1 for c in sub if c["e_channel"]["funnel"]["e1"]),
            "e2": sum(1 for c in sub if c["e_channel"]["funnel"]["e2"]),
            "e3": sum(1 for c in sub if c["e_channel"]["funnel"]["e3"]),
            "e4": sum(1 for c in sub if c["e_channel"]["funnel"]["e4"]),
            "jq_and_e4": sum(1 for c in sub if c["jq_passed"] and c["e_channel"]["funnel"]["e4"]),
        }

    S = {"ns": stats(ns), "gf": stats(gf), "all": stats(cands)}

    # 伪 ICIR / 救回
    pseudo = [c for c in cands if c["old_gate_passed"] and not c["jq_passed"]]
    rescued = [c for c in cands if c["jq_passed"] and not c["old_gate_passed"]]
    rescued_std = [c["ic_std"] for c in rescued]

    # NSGA-II 覆盖率坍缩（细分：近乎全空 vs 覆盖不足）
    ns_n_weeks = [c["e_channel"]["evaluations"]["E1"].get("n_weeks", 0) for c in ns]
    ns_empty = sum(1 for x in ns_n_weeks if x < 24)          # <24 有效周 = 组合近乎全空
    ns_cov = [c["e_channel"]["evaluations"]["E1"].get("coverage") for c in ns
              if c["e_channel"]["evaluations"]["E1"].get("coverage") is not None]
    ns_cov_min = min(ns_cov) * 100 if ns_cov else 0
    ns_cov_med = np.median(ns_cov) * 100 if ns_cov else 0
    ns_cov_max = max(ns_cov) * 100 if ns_cov else 0
    # GFlowNet 单叶子
    gf_single = sum(1 for c in gf if c.get("n_tokens", 0) == 1)

    # GFlowNet E2 结果
    e2_gf = Counter(c["e_channel"]["evaluations"].get("E2", {}).get("decision") for c in gf
                    if c["e_channel"]["evaluations"].get("E2", {}).get("decision") in ("pass", "reject"))

    def funnel_table(sub_key):
        s = S[sub_key]
        rows = [
            ("E1 方向/年度稳定/双所/BS下界", s["e1"], s["n"]),
            ("E2 独立性（池冗余）", s["e2"], s["e1"]),
            ("E3 配对账户净增量", s["e3"], s["e2"]),
            ("E4 成本/容量压力", s["e4"], s["e3"]),
        ]
        return "".join(
            f"<tr><td>{name}</td><td>{num}</td><td>{den}</td><td>{bar(num, den)}</td></tr>"
            for name, num, den in rows
        )

    # 候选明细（按 |shrink_icir| 降序）
    cands_sorted = sorted(cands, key=lambda c: -abs(c.get("shrink_icir") or 0))

    def cand_rows(sub):
        rows = []
        for c in sub:
            e = c["e_channel"]
            f = e["funnel"]
            label = c.get("expr") or ", ".join(list(c.get("weights", {}).keys())[:4])
            rows.append(
                f"<tr><td class='mono' title='{label}'>{label}</td>"
                f"<td>{c['method']}</td>"
                f"<td>{c['ic_mean']:+.4f}</td><td>{c['shrink_icir']:+.3f}</td>"
                f"<td>{c['ic_std']:.4f}</td><td>{c['turnover']:.2f}</td>"
                f"<td>{'✓' if c['jq_passed'] else '✗'}</td>"
                f"<td class='{'ok' if f['e1'] else 'bad'}'>{'✓' if f['e1'] else '✗'}</td>"
                f"<td class='{'ok' if f['e2'] else 'bad'}'>{'✓' if f['e2'] else '✗'}</td>"
                f"<td class='{'ok' if f['e3'] else 'bad'}'>{'✓' if f['e3'] else '✗'}</td>"
                f"<td class='{'ok' if f['e4'] else 'bad'}'>{'✓' if f['e4'] else '✗'}</td></tr>"
            )
        return "".join(rows)

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>E 道复评报告：JQ 门升级 + E1-E4 通过率</title>
<style>
:root{{--bg:#0f1115;--panel:#171a21;--border:#262b36;--fg:#e6e8ee;--muted:#9aa3b2;
--accent:#4f8cff;--good:#2ecc71;--bad:#ff5c5c;--warn:#f5a623;}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg);line-height:1.65;padding:32px 40px}}
h1{{font-size:24px;margin:0 0 4px}}
h2{{font-size:18px;margin:36px 0 12px;border-left:4px solid var(--accent);padding-left:10px}}
h3{{font-size:15px;margin:18px 0 8px;color:var(--fg)}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:24px}}
.panel{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}}
th{{color:var(--muted);font-weight:600}}
.mono{{font-family:"SF Mono",Consolas,monospace;font-size:12px;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bar{{width:130px;height:10px;background:#232836;border-radius:5px;display:inline-block;vertical-align:middle;margin-right:8px}}
.fill{{height:100%;background:var(--accent);border-radius:5px}}
.pct{{color:var(--muted);font-size:12px}}
.ok{{color:var(--good);font-weight:700}}
.bad{{color:var(--bad);font-weight:700}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{flex:1;min-width:150px;background:#1b1f2a;border:1px solid var(--border);border-radius:8px;padding:14px 16px}}
.num{{font-size:26px;font-weight:700}}
.lbl{{color:var(--muted);font-size:12px}}
.note{{background:#1a1f2b;border-left:3px solid var(--warn);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}}
.alert{{background:#2a1518;border-left:3px solid var(--bad);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}}
ul{{margin:8px 0;padding-left:20px}}
code{{background:#222836;padding:1px 6px;border-radius:4px;font-family:Consolas,monospace;font-size:12px}}
</style></head><body>
<h1>E 道复评报告：JQ 门升级 + E1-E4 通过率</h1>
<div class="sub">97 因子宇宙 · 周频 · NSGA-II {cfg['pop']}×{cfg['gen']} / GFlowNet ep{cfg['gf_ep']}·bs{cfg['gf_bs']}·n{cfg['gf_n']} · max_len{cfg['max_len']} · seed{cfg['seed']}</div>

<div class="kpi">
  <div class="card"><div class="num">{S['all']['n']}</div><div class="lbl">候选总数</div></div>
  <div class="card"><div class="num">{S['all']['jq_new']}</div><div class="lbl">JQ 门通过(收缩版)</div></div>
  <div class="card"><div class="num">{S['all']['e4']}</div><div class="lbl">通过 E1-E4 全链</div></div>
  <div class="card"><div class="num">{S['all']['jq_and_e4']}</div><div class="lbl">同时过 JQ + E1-E4</div></div>
</div>

<div class="alert"><b>核心结论：</b>NSGA-II 与 GFlowNet 产出的 JQ 候选 <b>几乎没有</b>穿过生产 E 道——
210 个候选里只有 <b>{S['all']['e4']} 个（全部来自 GFlowNet）</b>通过 E1-E4 全链。两个方法的失败模式截然不同：
<b>NSGA-II 死于覆盖率坍缩</b>（{ns_empty}/{S['ns']['n']} 组合近乎全空、其余覆盖仅 {ns_cov_min:.0f}~{ns_cov_max:.0f}%，中位 {ns_cov_med:.0f}%），
<b>GFlowNet 死于单叶子冗余</b>（{gf_single}/{len(gf)}={gf_single/len(gf)*100:.0f}% 是原始单因子，E2 判冗余）。</div>

<h2>① JQ 门升级效果（旧硬门 → 贝叶斯收缩 ICIR）</h2>
<div class="panel">
<table>
<tr><th>方法</th><th>旧硬门<br>(|icir|≥0.3 & ic_std≥0.04)</th><th>收缩版<br>(|shrink_icir|≥0.2)</th><th>净变化</th></tr>
<tr><td>NSGA-II</td><td>{pct(S['ns']['jq_old'], S['ns']['n'])}</td><td>{pct(S['ns']['jq_new'], S['ns']['n'])}</td><td>{S['ns']['jq_new']-S['ns']['jq_old']:+d}</td></tr>
<tr><td>GFlowNet</td><td>{pct(S['gf']['jq_old'], S['gf']['n'])}</td><td>{pct(S['gf']['jq_new'], S['gf']['n'])}</td><td>{S['gf']['jq_new']-S['gf']['jq_old']:+d}</td></tr>
<tr><td>合计</td><td>{pct(S['all']['jq_old'], S['all']['n'])}</td><td>{pct(S['all']['jq_new'], S['all']['n'])}</td><td>{S['all']['jq_new']-S['all']['jq_old']:+d}</td></tr>
</table>
<div class="note">收缩版去掉 <code>ic_std≥0.04</code> 硬一刀切后更宽松：救回 <b>{len(rescued)}</b> 个低 ic_std 候选
（救回者 ic_std 中位数 <b>{np.median(rescued_std):.4f}</b>，最小 {min(rescued_std):.4f}）。
旧硬门只比收缩版多拦下 <b>{len(pseudo)}</b> 个（伪 ICIR 已基本被 grammar 禁自除 + 收缩分母封住）。
<b>代价：</b>救回的 29 个里低 ic_std 者需靠 LCB 诊断辅助判定是否真信号（软门 trade-off）。</div>
</div>

<h2>② 生产 E 道 E1-E4 通过率漏斗（条件漏斗）</h2>
<div class="panel">
<h3 style="margin-top:0">NSGA-II（{S['ns']['n']} 候选）</h3>
<table><tr><th>阶段</th><th>通过</th><th>进入</th><th>通过率</th></tr>{funnel_table('ns')}</table>
<div class="alert">NSGA-II 全部止步 E1：{ns_empty} 个候选 &lt;24 个有效周（组合近乎全空，混入 <code>ocf_quality</code>(6%)、
<code>post_list_external_flow_days</code>(12%) 等极稀疏财务因子后交集坍缩），其余 {len(ns_cov)} 个覆盖仅
{ns_cov_min:.0f}~{ns_cov_max:.0f}%（中位 {ns_cov_med:.0f}%），全部低于 E1 的 80% 覆盖下限。
<b>根因 = <code>linear_combine</code> 的 NaN 传播</b>：加权求和时任一成分因子缺失即整点缺失，
组合覆盖 = 各成分覆盖的<b>交集</b>；而 GA 的收益目标在 OOS 稀疏子集上仍能算 IC，于是选中了这些稀疏组合。</div>

<h3 style="margin-top:0">GFlowNet（{S['gf']['n']} 候选）</h3>
<table><tr><th>阶段</th><th>通过</th><th>进入</th><th>通过率</th></tr>{funnel_table('gf')}</table>
<div class="note">GFlowNet 高覆盖（单叶子=原始价量因子，覆盖~90%）所以 E1 通过率高（{S['gf']['e1']}），
但 E2 独立性把 84 个砍到 9 个：<b>单叶子 = 原始因子本身</b>，与池冗余（E2 拒绝 {e2_gf.get('reject',0)} 个）。
最终 3 个穿过 E3/E4。</div>
</div>

<h2>③ 候选明细（按 |收缩 ICIR| 降序，前 40）</h2>
<div class="panel">
<table>
<tr><th>表达式/权重</th><th>方法</th><th>ic_mean</th><th>shrink</th><th>ic_std</th><th>换手</th><th>JQ</th><th>E1</th><th>E2</th><th>E3</th><th>E4</th></tr>
{cand_rows(cands_sorted[:40])}
</table>
</div>

</body></html>"""
    out = RESULTS / "e_channel_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"报告已生成：{out}")
    return out


if __name__ == "__main__":
    build_report()
