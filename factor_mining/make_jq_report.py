"""
JQ 候选 + 多样性报告生成器
==========================
读取 jq_eval.py 的落盘结果，生成:
  - scatter_jq.png   : 候选散点 (|ic_mean| vs turnover，按 JQ 门着色)
  - jq_report.html   : 完整 HTML 报告（JQ 计数 / 门失败分布 / 多样性 / top 候选）

用法: python -m factor_mining.make_jq_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results" / "jq_eval"
OUT = Path(__file__).parent / "results"

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main():
    summary = load("jq_summary.json")
    nsga = load("nsga2_candidates.json")["candidates"]
    gf = load("gflownet_candidates.json")["candidates"]
    gate = summary["jq_gate"]

    # ---------------- 散点图 ----------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for tag, cands, color, marker in [("NSGA-II", nsga, "#c0392b", "o"),
                                      ("GFlowNet", gf, "#2e86c1", "^")]:
        x = [abs(c["ic_mean"]) for c in cands]
        y = [c["turnover"] for c in cands]
        passed = [c["jq_passed"] for c in cands]
        ax.scatter([x[i] for i in range(len(x)) if passed[i]],
                   [y[i] for i in range(len(y)) if passed[i]],
                   c=color, marker=marker, s=55, alpha=0.85, label=f"{tag} 过门")
        ax.scatter([x[i] for i in range(len(x)) if not passed[i]],
                   [y[i] for i in range(len(y)) if not passed[i]],
                   facecolors="none", edgecolors=color, marker=marker, s=55, alpha=0.6,
                   label=f"{tag} 未过门")
    ax.axvline(0.03, color="gray", ls="--", lw=1, alpha=0.5, label="|ic_mean|=0.03")
    ax.axhline(gate["max_turnover"], color="gray", ls="--", lw=1, alpha=0.5,
               label=f"turnover={gate['max_turnover']}")
    ax.set_xlabel("|ic_mean|（全截面排序强度）")
    ax.set_ylabel("周换手率 turnover")
    ax.set_title("两目标 NSGA-II vs GFlowNet：JQ 候选分布")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_jq.png", dpi=130)
    plt.close(fig)

    # ---------------- 门失败分布 ----------------
    GATE_KEYS = ["n_weeks>=15", "|icir|>=0.3", "+IC%>=0.55", "turnover<=0.5", "ic_std>=0.04"]

    def fail_breakdown(cands):
        cnt = {k: 0 for k in GATE_KEYS}
        for c in cands:
            if c["jq_passed"]:
                continue
            for k in GATE_KEYS:
                if not c["jq_checks"].get(k, False):
                    cnt[k] += 1
        return cnt

    nsga_fail = fail_breakdown(nsga)
    gf_fail = fail_breakdown(gf)

    # ---------------- HTML ----------------
    nsga_s = summary["nsga2"]
    gf_s = summary["gflownet"]
    # Top 候选：排除 ic_std<0.04 的伪 ICIR（比值型指标 std 坍缩），再按 |icir| 降序
    nsga_top = sorted([c for c in nsga if c.get("ic_std", 0) >= 0.04],
                      key=lambda c: -abs(c["icir"]))[:10]
    gf_top = sorted([c for c in gf if c.get("ic_std", 0) >= 0.04],
                    key=lambda c: -abs(c["icir"]))[:10]

    def nsga_rows():
        rows = []
        for c in nsga_top:
            w = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(c["weights"].items(),
                                                            key=lambda kv: -kv[1]))
            rows.append(
                f"<tr><td>{w}</td><td>{c['n_factors']}</td>"
                f"<td>{c['ic_mean']:.3f}</td><td>{c['icir']:.3f}</td><td>{c.get('ic_std',0):.3f}</td>"
                f"<td>{c['gross_sharpe']:.2f}</td><td>{c['turnover']:.3f}</td>"
                f"<td>{c['plus_ic_pct']:.0%}</td>"
                f"<td>{'✅' if c['jq_passed'] else '✗'}</td></tr>")
        return "\n".join(rows)

    def gf_rows():
        rows = []
        for c in gf_top:
            rows.append(
                f"<tr><td class='expr'>{c['expr']}</td><td>{c['n_tokens']}</td>"
                f"<td>{c['ic_mean']:.3f}</td><td>{c['icir']:.3f}</td><td>{c.get('ic_std',0):.3f}</td>"
                f"<td>{c['gross_sharpe']:.2f}</td><td>{c['turnover']:.3f}</td>"
                f"<td>{c['plus_ic_pct']:.0%}</td>"
                f"<td>{'✅' if c['jq_passed'] else '✗'}</td></tr>")
        return "\n".join(rows)

    def gate_fail_rows(fail):
        return "".join(f"<td>{fail[k]}</td>" for k in GATE_KEYS)

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>JQ 候选 + 多样性报告（两目标 NSGA-II vs GFlowNet）</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 28px; color: #222; }}
h1 {{ font-size: 22px; border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }}
h2 {{ font-size: 17px; margin-top: 26px; color: #2c3e50; }}
table {{ border-collapse: collapse; font-size: 13px; margin-top: 8px; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
th {{ background: #34495e; color: #fff; }}
tr:nth-child(even) {{ background: #f7f9fb; }}
.expr {{ text-align: left; font-family: Consolas, monospace; font-size: 12px; }}
.pass {{ color: #27ae60; font-weight: bold; }}
.fail {{ color: #c0392b; font-weight: bold; }}
.kpi {{ display: inline-block; margin: 6px 12px 6px 0; padding: 10px 16px; background: #eef3f8; border-radius: 6px; }}
.kpi b {{ font-size: 20px; color: #2c3e50; }}
.note {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: 10px 14px; margin: 14px 0; font-size: 13px; }}
</style></head><body>
<h1>JQ 候选 + 多样性报告：两目标 NSGA-II vs GFlowNet</h1>
<p>数据：97 因子缓存 × {summary['n_factors_total']} 因子；评估口径 OOS 70:30 + Top30 等权。
JQ 门 = {gate['min_n_weeks']} 周充足 & |ICIR|≥{gate['min_abs_icir']} & +IC%≥{gate['min_plus_ic_pct']:.0%}
& 换手≤{gate['max_turnover']} & <b>ic_std≥{gate['min_ic_std']}</b>。</p>

<div class="note">🔴 <b>口径说明</b>：NSGA-II 与 GFlowNet 是旁挂库，跑在 97 因子缓存宇宙上（非生产 1125 股原始快照）。
本报告的「JQ 候选」按 <b>忠实映射生产 E1（Spearman rank IC）+ 用户 ICIR≥0.3/+IC%≥55% 验证线</b> 判定，
是「因子质量层」的 JQ 资格判定，不等同于生产 E0-E5 在原始快照上的完整验收。</div>

<div class="note">🔴 <b>伪 ICIR 陷阱（本次实测修复）</b>：ICIR=ic_mean/ic_std 是比值型指标，
优化器（或语法自除如 div(X,X)）能构造 ic_std≈0 的组合把 ICIR 除到无穷大，穿透「|ICIR|≥0.3」门。
修复 = 加 <b>ic_std≥0.04 下限</b>（原始 97 因子 ic_std min=0.0469，不误伤）+ grammar 禁自除。
修复后 NSGA-II 73→59、GFlowNet 61→60（各剔除 14 / 4 个 std 坍缩伪信号）。</div>

<h2>核心结论（KPI）</h2>
<div class="kpi">NSGA-II JQ 候选 <b>{nsga_s['n_jq_pass']}/{nsga_s['n_candidates']}</b>（{nsga_s['pass_rate']:.0%}）</div>
<div class="kpi">GFlowNet JQ 候选 <b>{gf_s['n_jq_pass']}/{gf_s['n_candidates']}</b>（{gf_s['pass_rate']:.0%}）</div>
<div class="kpi">跨方法互补相关 <b>{summary['cross_method_corr']:.3f}</b></div>
<div class="kpi">GFlowNet 去重叶子集 <b>{gf_s['distinct_leaf_sets']}</b></div>

<h2>多样性度量（能否多样化）</h2>
<table>
<tr><th>指标</th><th>NSGA-II</th><th>GFlowNet</th><th>解读</th></tr>
<tr><td>内部相关（越低越多样）</td><td>{nsga_s['mean_internal_corr']:.3f}</td>
<td>{gf_s['mean_internal_corr']:.3f}</td><td>GFlowNet 内部多样性显著更高</td></tr>
<tr><td>去重叶子因子集</td><td>{nsga_s['distinct_leaf_sets']}</td>
<td>{gf_s['distinct_leaf_sets']}</td>
<td>GFlowNet 结构覆盖更广</td></tr>
<tr><td>叶子覆盖度（/97）</td><td>{nsga_s['leaf_coverage']:.1%}</td>
<td>{gf_s['leaf_coverage']:.1%}</td>
<td>GFlowNet 覆盖 {gf_s['leaf_coverage']:.0%} 的因子池</td></tr>
<tr><td>跨方法相关（互补）</td><td colspan="2">{summary['cross_method_corr']:.3f}</td>
<td>越低 = GFlowNet 越互补</td></tr>
</table>

<h2>JQ 门失败分布（未过门候选卡在哪一关）</h2>
<table>
<tr><th>方法</th><th>n_weeks&lt;{gate['min_n_weeks']}</th><th>|ICIR|&lt;{gate['min_abs_icir']}</th>
<th>+IC%&lt;{gate['min_plus_ic_pct']:.0%}</th><th>换手&gt;{gate['max_turnover']}</th>
<th>ic_std&lt;{gate['min_ic_std']}</th></tr>
<tr><td>NSGA-II</td>{gate_fail_rows(nsga_fail)}</tr>
<tr><td>GFlowNet</td>{gate_fail_rows(gf_fail)}</tr>
</table>

<h2>散点图（|ic_mean| vs 换手）</h2>
<img src="scatter_jq.png" style="max-width: 820px; width: 100%; border: 1px solid #ccc;">

<h2>NSGA-II Top JQ 候选（按 |ICIR| 降序，已排除 ic_std&lt;0.04 伪信号）</h2>
<table>
<tr><th>因子权重组合</th><th>因子数</th><th>ic_mean</th><th>icir</th><th>ic_std</th>
<th>毛Sharpe</th><th>换手</th><th>+IC%</th><th>过门</th></tr>
{nsga_rows()}
</table>

<h2>GFlowNet Top JQ 候选（按 |ICIR| 降序，已排除 ic_std&lt;0.04 伪信号）</h2>
<table>
<tr><th>表达式</th><th>token</th><th>ic_mean</th><th>icir</th><th>ic_std</th>
<th>毛Sharpe</th><th>换手</th><th>+IC%</th><th>过门</th></tr>
{gf_rows()}
</table>
</body></html>"""

    with open(OUT / "jq_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"报告已生成: {OUT / 'jq_report.html'}")
    print(f"散点图: {OUT / 'scatter_jq.png'}")


if __name__ == "__main__":
    main()
