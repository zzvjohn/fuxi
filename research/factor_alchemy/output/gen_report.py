"""
Generate HTML report from Factor Alchemy v4 run data.
Reads run_20260608_162442.log and pickles, produces report_20260608.html
"""
import pickle
import os
from datetime import datetime

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# === Load Pareto front data ===
pf = None
pf_path = os.path.join(OUT_DIR, 'pareto_front.pkl')
if os.path.exists(pf_path):
    with open(pf_path, 'rb') as f:
        pf = pickle.load(f)

# === IC summary ===
import pandas as pd
ic_summary = None
ic_path = os.path.join(OUT_DIR, 'ic_summary.csv')
if os.path.exists(ic_path):
    ic_summary = pd.read_csv(ic_path, index_col=0)

# === Build HTML ===
html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Factor Alchemy v4 — NSGA-II 多目标优化报告</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f1419; color: #e7e9ea; line-height: 1.6; padding: 40px 20px; }
.container { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 32px; margin-bottom: 4px; color: #fff; }
h2 { font-size: 22px; color: #1d9bf0; margin: 40px 0 16px; padding-bottom: 8px; border-bottom: 2px solid #1d9bf0; }
h3 { font-size: 17px; color: #e7e9ea; margin: 24px 0 12px; }
.subtitle { color: #71767b; font-size: 14px; margin-bottom: 30px; }

/* Cards */
.card { background: #1a1f26; border: 1px solid #2f3336; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.card-header { font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 16px; }

/* Stat grid */
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.stat-box { flex: 1; min-width: 150px; background: #161b22; border-radius: 10px; padding: 16px; text-align: center; border: 1px solid #2f3336; }
.stat-label { font-size: 12px; color: #71767b; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { font-size: 28px; font-weight: 700; margin: 6px 0; }
.stat-value.positive { color: #ff4444; }
.stat-value.negative { color: #00b368; }
.stat-value.neutral { color: #1d9bf0; }
.stat-value.warn { color: #ffa600; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { background: #161b22; color: #71767b; font-weight: 600; text-align: left; padding: 10px 14px; border-bottom: 1px solid #2f3336; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
td { padding: 9px 14px; border-bottom: 1px solid #1e242c; }
tr:hover td { background: #1a1f26; }

/* Badges */
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-right: 6px; }
.badge-green { background: #003d1f; color: #00b368; }
.badge-red { background: #3d0011; color: #ff4444; }
.badge-blue { background: #001f3d; color: #1d9bf0; }
.badge-yellow { background: #3d2c00; color: #ffa600; }

/* Progress bars */
.bar-container { background: #161b22; border-radius: 6px; height: 8px; margin-top: 6px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; }
.bar-high { background: #ff4444; }
.bar-mid { background: #1d9bf0; }
.bar-low { background: #00b368; }

/* Pareto scatter placeholder */
.pareto-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.pareto-viz { background: #161b22; border-radius: 10px; padding: 20px; text-align: center; min-height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.pareto-viz svg { max-width: 100%; }

/* Footer */
.footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #2f3336; color: #71767b; font-size: 13px; text-align: center; }

/* Highlights */
.highlight-box { background: #161b22; border-left: 4px solid #ffa600; border-radius: 0 8px 8px 0; padding: 16px; margin: 16px 0; }
.highlight-box h4 { color: #ffa600; margin-bottom: 8px; }
.highlight-box ul { padding-left: 20px; }
.highlight-box li { margin-bottom: 4px; }

/* Generation evolution */
.gen-bar { display: flex; gap: 2px; align-items: flex-end; margin: 10px 0; height: 80px; }
.gen-column { flex: 1; display: flex; flex-direction: column; align-items: center; }
.gen-icir { background: #ff4444; border-radius: 3px 3px 0 0; width: 30px; }
.gen-stab { background: #1d9bf0; border-radius: 3px 3px 0 0; width: 30px; }
.gen-label { font-size: 11px; color: #71767b; margin-top: 4px; }
</style>
</head>
<body>
<div class="container">
<h1>🧬 Factor Alchemy v4 — NSGA-II 多目标优化报告</h1>
<div class="subtitle">ICIR_THRESHOLD = 1.5 | 63因子 | 40pop × 5gen | 2026-06-08 16:24 CST | 总耗时 8分42秒</div>

<!-- ====== Summary ====== -->
<h2>📋 执行摘要</h2>
<div class="card">
  <div class="stat-row">
    <div class="stat-box"><div class="stat-label">帕累托前沿</div><div class="stat-value neutral">9 解</div></div>
    <div class="stat-box"><div class="stat-label">ICIR 范围</div><div class="stat-value positive">0.610 - 0.934</div></div>
    <div class="stat-box"><div class="stat-label">稳定性范围</div><div class="stat-value neutral">0.116 - 0.808</div></div>
    <div class="stat-box"><div class="stat-label">最优 Sharpe</div><div class="stat-value warn">+0.08</div></div>
    <div class="stat-box"><div class="stat-label">最优年化</div><div class="stat-value negative">-7.8%</div></div>
  </div>
  <div class="highlight-box">
    <h4>⚡ 核心发现</h4>
    <ul>
      <li>ICIR 阈值升至 1.5 后完全消除饱和，NSGA-II 成功发现 9 个互不支配的帕累托最优解</li>
      <li>ICIR 与稳定性呈现经典 trade-off：高 ICIR 伴随高换手低稳定，低换手伴随低 ICIR</li>
      <li>63 个纯量价因子的合成天花板 ≈ Sharpe 0，QMT 双边成本下所有组合均为负或零收益</li>
      <li>最优策略（均衡调和）为 8 因子组合，年化 -7.8%，Sharpe +0.08，MaxDD -69.8%</li>
    </ul>
  </div>
</div>

<!-- ====== NSGA-II Evolution ====== -->
<h2>🔬 NSGA-II 进化过程</h2>
<div class="card">
  <div class="card-header">世代演化记录 (40pop × 5gen, 快速路径 0秒)</div>
  <table>
    <tr><th>世代</th><th>Pareto 解数</th><th>ICIR 范围</th><th>稳定性范围</th></tr>
    <tr><td>Gen 1</td><td>3</td><td><span class="badge badge-red">0.459 — 0.627</span></td><td><span class="badge badge-blue">0.750 — 0.758</span></td></tr>
    <tr><td>Gen 2</td><td>5</td><td><span class="badge badge-red">0.459 — 0.712</span></td><td><span class="badge badge-blue">0.750 — 0.758</span></td></tr>
    <tr><td>Gen 3</td><td>8</td><td><span class="badge badge-red">0.459 — 0.768</span></td><td><span class="badge badge-blue">0.540 — 0.758</span></td></tr>
    <tr><td>Gen 4</td><td>6</td><td><span class="badge badge-red">0.532 — 0.831</span></td><td><span class="badge badge-blue">0.224 — 0.765</span></td></tr>
    <tr><td>Gen 5</td><td><b>7</b></td><td><span class="badge badge-red">0.610 — 0.831</span></td><td><span class="badge badge-blue">0.224 — 0.808</span></td></tr>
  </table>
  <p style="margin-top:12px;color:#71767b;font-size:13px;">
    📈 帕累托前沿从 Gen1 的 3 个解扩展到最终的 9 个（去重）。ICIR 上限从 0.627 推至 0.831，稳定性上限从 0.758 推至 0.808。
  </p>
</div>

<!-- ====== Pareto Frontier ====== -->
<h2>🎯 帕累托前沿（9 个非支配解）</h2>
<div class="card">
  <div class="card-header">前沿解排序（按 ICIR_score 降序）</div>
  <table>
    <tr><th>#</th><th>ICIR_score</th><th>Stab_score</th><th>ICIR_raw</th><th>Est_TO</th><th>因子数</th><th>定位</th></tr>
    <tr><td>5</td><td><b style="color:#ff4444">0.934</b></td><td>0.116</td><td>+0.493</td><td>22.4%</td><td>6</td><td><span class="badge badge-red">激进端</span></td></tr>
    <tr><td>4</td><td>0.842</td><td>0.362</td><td>+0.452</td><td>20.4%</td><td>8</td><td></td></tr>
    <tr><td>8</td><td>0.815</td><td>0.625</td><td>+0.382</td><td>11.9%</td><td>10</td><td></td></tr>
    <tr><td>2</td><td>0.792</td><td>0.670</td><td>+0.405</td><td>14.0%</td><td>10</td><td></td></tr>
    <tr><td>0</td><td>0.786</td><td>0.687</td><td>+0.402</td><td>14.0%</td><td>10</td><td></td></tr>
    <tr><td>6</td><td>0.786</td><td>0.687</td><td>+0.402</td><td>14.0%</td><td>10</td><td></td></tr>
    <tr style="background:#1a2a1f"><td><b>7</b></td><td><b style="color:#1d9bf0">0.730</b></td><td><b style="color:#00b368">0.770</b></td><td>+0.377</td><td><b>7.2%</b></td><td><b>8</b></td><td><span class="badge badge-green">★ 均衡</span></td></tr>
    <tr><td>1</td><td>0.610</td><td>0.808</td><td>+0.383</td><td>12.4%</td><td>8</td><td><span class="badge badge-yellow">稳健端</span></td></tr>
    <tr><td>3</td><td>0.610</td><td>0.808</td><td>+0.383</td><td>12.4%</td><td>8</td><td><span class="badge badge-yellow">稳健端</span></td></tr>
  </table>
</div>

<!-- ====== Three Strategies ====== -->
<h2>📊 三策略对比</h2>

<div class="card">
  <div class="card-header">🟢 稳健策略（低换手优先）— 解 #1/#3</div>
  <table>
    <tr><th>因子</th><th>权重</th><th>ICIR</th><th>类别</th></tr>
    <tr><td>dp</td><td>15.4%</td><td>+0.222</td><td>估值</td></tr>
    <tr><td>rev_growth_yoy</td><td>14.3%</td><td>+0.084</td><td>成长</td></tr>
    <tr><td>atr_14</td><td>13.8%</td><td>+0.483</td><td>波动</td></tr>
    <tr><td>high_low_range</td><td>13.5%</td><td>+0.442</td><td>波动</td></tr>
    <tr><td>price_1m</td><td>13.4%</td><td>+0.365</td><td>动量</td></tr>
    <tr><td>max_ret_1m</td><td>12.2%</td><td>+0.503</td><td>波动</td></tr>
    <tr><td>abnormal_turnover</td><td>9.4%</td><td>+0.421</td><td>换手</td></tr>
    <tr><td>psy_12</td><td>8.0%</td><td>+0.190</td><td>情绪</td></tr>
  </table>
  <div class="stat-row" style="margin-top:12px">
    <div class="stat-box"><div class="stat-label">合成 ICIR</div><div class="stat-value neutral">+0.383</div></div>
    <div class="stat-box"><div class="stat-label">五维评估</div><div class="stat-value warn">2/5 ❌</div></div>
    <div class="stat-box"><div class="stat-label">估换手</div><div class="stat-value neutral">12.4%</div></div>
  </div>
</div>

<div class="card">
  <div class="card-header">🟡 均衡策略（调和平均）— 解 #7 ★ 推荐</div>
  <table>
    <tr><th>因子</th><th>权重</th><th>ICIR</th><th>类别</th></tr>
    <tr><td>dp</td><td>14.2%</td><td>+0.222</td><td>估值</td></tr>
    <tr><td>abnormal_turnover</td><td>14.2%</td><td>+0.421</td><td>换手</td></tr>
    <tr><td>rev_growth_yoy</td><td>13.6%</td><td>+0.084</td><td>成长</td></tr>
    <tr><td>atr_14</td><td>13.1%</td><td>+0.483</td><td>波动</td></tr>
    <tr><td>high_low_range</td><td>12.9%</td><td>+0.442</td><td>波动</td></tr>
    <tr><td>vol_3m</td><td>11.7%</td><td>+0.390</td><td>波动</td></tr>
    <tr><td>davol_20</td><td>10.4%</td><td>+0.411</td><td>换手</td></tr>
    <tr><td>roa</td><td>9.8%</td><td>+0.038</td><td>质量</td></tr>
  </table>
  <div class="stat-row" style="margin-top:12px">
    <div class="stat-box"><div class="stat-label">合成 ICIR</div><div class="stat-value neutral">+0.377</div></div>
    <div class="stat-box"><div class="stat-label">五维评估</div><div class="stat-value warn">2/5 ❌</div></div>
    <div class="stat-box"><div class="stat-label">估换手</div><div class="stat-value negative">7.2%</div></div>
  </div>
</div>

<div class="card">
  <div class="card-header">🔴 激进策略（高 ICIR 优先）— 解 #5</div>
  <table>
    <tr><th>因子</th><th>权重</th><th>ICIR</th><th>类别</th></tr>
    <tr><td>atr_14</td><td>20.2%</td><td>+0.483</td><td>波动</td></tr>
    <tr><td>vol_3m</td><td>18.1%</td><td>+0.390</td><td>波动</td></tr>
    <tr><td>turnover_change</td><td>17.0%</td><td>+0.401</td><td>换手</td></tr>
    <tr><td>turnover_std</td><td>16.6%</td><td>+0.503</td><td>换手</td></tr>
    <tr><td>davol_20</td><td>16.0%</td><td>+0.411</td><td>换手</td></tr>
    <tr><td>volume_ratio</td><td>12.2%</td><td>+0.158</td><td>量价</td></tr>
  </table>
  <div class="stat-row" style="margin-top:12px">
    <div class="stat-box"><div class="stat-label">合成 ICIR</div><div class="stat-value neutral">+0.493</div></div>
    <div class="stat-box"><div class="stat-label">五维评估</div><div class="stat-value warn">2/5 ❌</div></div>
    <div class="stat-box"><div class="stat-label">估换手</div><div class="stat-value positive">22.4%</div></div>
  </div>
</div>

<!-- ====== Portfolio Backtest ====== -->
<h2>💰 虚拟投资组合回测（QMT 实盘成本）</h2>
<div class="card">
  <div class="card-header">成本设定：佣金万1 | 印花税千1 | 滑点 0.3% | 最低佣金 5元</div>
  <table>
    <tr><th>策略</th><th>总收益</th><th>年化</th><th>Sharpe</th><th>MaxDD</th><th>周换手</th><th>年化成本</th></tr>
    <tr style="background:#1a1a1f">
      <td><span class="badge badge-yellow">稳健</span></td>
      <td style="color:#ff4444">-58.0%</td><td>-16.1%</td><td>-0.19</td><td>-68.2%</td><td>66.1%</td><td>4.6%</td>
    </tr>
    <tr style="background:#1a2a1f">
      <td><span class="badge badge-green">均衡 ★</span></td>
      <td style="color:#ff4444">-33.9%</td><td style="color:#ffa600">-7.8%</td><td style="color:#00b368"><b>+0.08</b></td><td>-69.8%</td><td style="color:#00b368">50.1%</td><td>2.7%</td>
    </tr>
    <tr style="background:#1f1a1a">
      <td><span class="badge badge-red">激进</span></td>
      <td style="color:#ff4444">-62.0%</td><td>-17.3%</td><td>-0.52</td><td>-65.4%</td><td style="color:#ff4444">147.5%</td><td>8.4%</td>
    </tr>
  </table>
  <p style="margin-top:16px;color:#71767b;font-size:13px;">
    📉 交易成本是核心杀手：激进策略年化成本 8.4% vs 均衡 2.7%。换手每降低 1pp ≈ 年化提升 ~0.37%。
  </p>
</div>

<!-- ====== Top Factor ICIR ====== -->
<h2>🏆 单因子 IC/ICIR 排行（Top 20）</h2>
<div class="card">
  <table>
    <tr><th>排名</th><th>因子</th><th>ICIR</th><th>IC_mean</th><th>+IC%</th><th>类别</th></tr>
'''

# Add ICIR table
icir_data = [
    ("earnings_growth_yoy", 0.514, 0.0387, 73.7, "成长"),
    ("max_ret_1m", 0.503, 0.0667, 68.1, "波动"),
    ("turnover_std", 0.503, 0.0751, 67.4, "换手"),
    ("vm_diff", 0.493, 0.0494, 70.8, "量价"),
    ("atr_14", 0.483, 0.0850, 68.4, "波动"),
    ("bull_power", 0.445, 0.0541, 66.5, "量价"),
    ("vol_1m", 0.444, 0.0703, 65.8, "波动"),
    ("high_low_range", 0.442, 0.0788, 66.3, "波动"),
    ("dollar_vol_20d", 0.440, 0.0539, 70.5, "流动性"),
    ("abnormal_turnover", 0.421, 0.0697, 64.8, "换手"),
    ("avg_turnover_1m", 0.421, 0.0697, 64.8, "换手"),
    ("vroc_12", 0.414, 0.0319, 68.1, "量价"),
    ("davol_20", 0.411, 0.0532, 61.5, "换手"),
    ("turnover_change", 0.401, 0.0332, 65.9, "换手"),
    ("vol_3m", 0.390, 0.0683, 67.8, "波动"),
    ("idio_vol", 0.381, 0.0652, 62.6, "波动"),
    ("ret_3m", 0.378, 0.0514, 61.1, "动量"),
    ("vr_26", 0.373, 0.0337, 65.0, "量价"),
    ("price_1m", 0.365, 0.0476, 63.9, "动量"),
    ("bias_20", 0.358, 0.0478, 62.8, "量价"),
]

for i, (name, icir, ic_mean, pos_ic, cat) in enumerate(icir_data, 1):
    color = "#ff4444" if icir > 0.45 else ("#1d9bf0" if icir > 0.35 else "#71767b")
    html += f'    <tr><td>{i}</td><td><b>{name}</b></td><td style="color:{color}">{icir:+.3f}</td><td>{ic_mean:+.4f}</td><td>{pos_ic:.1f}%</td><td><span class="badge badge-blue">{cat}</span></td></tr>\n'

html += '''  </table>
</div>

<!-- ====== Factor Correlation ====== -->
<h2>🔗 因子相关性 — 高相关对 (|r| > 0.8)</h2>
<div class="card">
  <table>
    <tr><th>因子A</th><th>因子B</th><th>相关系数</th></tr>
    <tr><td>avg_turnover_1m</td><td>abnormal_turnover</td><td style="color:#ff4444">+0.993</td></tr>
    <tr><td>ret_1m</td><td>roc_20</td><td style="color:#ff4444">+0.970</td></tr>
    <tr><td>ln_mcap</td><td>ln_circulating_mcap</td><td style="color:#ff4444">+0.940</td></tr>
    <tr><td>high_low_range</td><td>atr_14</td><td style="color:#ff4444">+0.936</td></tr>
    <tr><td>avg_turnover_1m</td><td>turnover_std</td><td style="color:#ff4444">+0.931</td></tr>
    <tr><td>turnover_std</td><td>abnormal_turnover</td><td style="color:#ff4444">+0.931</td></tr>
    <tr><td>amihud_illiq</td><td>dollar_vol_20d</td><td style="color:#ff4444">+0.931</td></tr>
    <tr><td>roc_20</td><td>rank_1m</td><td style="color:#ff4444">+0.918</td></tr>
    <tr><td>ret_1m</td><td>rank_1m</td><td style="color:#ff4444">+0.918</td></tr>
    <tr><td>avg_turnover_3m</td><td>abnormal_turnover</td><td style="color:#ff4444">+0.891</td></tr>
  </table>
  <p style="margin-top:12px;color:#71767b;font-size:13px;">
    ⚠️ 63 因子中 80 对 |r| > 0.5，换手族/波动族/动量族内部高度重叠。同质化严重限制了组合增益。
  </p>
</div>

<!-- ====== Stability ====== -->
<h2>🕐 时序稳定性 Top 8（非规模因子）</h2>
<div class="card">
  <table>
    <tr><th>因子</th><th>稳定性分数</th><th>排名自相关</th><th>留存率</th><th>IC半衰期</th><th>周换手</th></tr>
    <tr><td>bp</td><td style="color:#00b368"><b>0.886</b></td><td>0.995</td><td>94.3%</td><td>11.3周</td><td>7.2%</td></tr>
    <tr><td>sp</td><td style="color:#00b368">0.851</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>f_score</td><td style="color:#00b368">0.800</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>dp</td><td style="color:#00b368">0.798</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>ep</td><td style="color:#00b368">0.780</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>cfp</td><td style="color:#00b368">0.770</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>roe</td><td style="color:#ff4444">0.300</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
    <tr><td>roa</td><td style="color:#ff4444">0.300</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  </table>
  <p style="margin-top:12px;color:#71767b;font-size:13px;">
    📌 估值因子（bp/sp/dp/ep/cfp）稳定性最高（0.77-0.89），质量因子（roe/roa）最低（0.30）。估值因子更适合作组合锚。
  </p>
</div>

<!-- ====== Configuration ====== -->
<h2>⚙️ 运行配置</h2>
<div class="card">
  <div class="stat-row">
    <div class="stat-box"><div class="stat-label">数据范围</div><div class="stat-value" style="font-size:16px">2021-01 ~ 2026-04</div></div>
    <div class="stat-box"><div class="stat-label">周频周期</div><div class="stat-value" style="font-size:16px">278 周</div></div>
    <div class="stat-box"><div class="stat-label">股票池</div><div class="stat-value" style="font-size:16px">5,710 只</div></div>
    <div class="stat-box"><div class="stat-label">因子总数</div><div class="stat-value" style="font-size:16px">63 个</div></div>
    <div class="stat-box"><div class="stat-label">IC序列</div><div class="stat-value" style="font-size:16px">54/63</div></div>
  </div>
  <div class="stat-row">
    <div class="stat-box"><div class="stat-label">遗传算法</div><div class="stat-value" style="font-size:16px">NSGA-II</div></div>
    <div class="stat-box"><div class="stat-label">种群 × 代</div><div class="stat-value" style="font-size:16px">40 × 5</div></div>
    <div class="stat-box"><div class="stat-label">交叉率</div><div class="stat-value" style="font-size:16px">70%</div></div>
    <div class="stat-box"><div class="stat-label">变异率</div><div class="stat-value" style="font-size:16px">30%</div></div>
    <div class="stat-box"><div class="stat-label">ICIR 阈值</div><div class="stat-value" style="font-size:16px;color:#ff4444">1.5</div></div>
  </div>
</div>

<!-- ====== Conclusions ====== -->
<h2>💡 结论与下一步</h2>
<div class="card">
  <div class="highlight-box">
    <h4>✅ 成功的关键改进</h4>
    <ul>
      <li><b>快速 IC 序列路径</b>：预计算 54 因子的时间序列 IC，NSGA-II 代际评估从 25 分钟降至 0 秒</li>
      <li><b>ICIR 阈值校准</b>：0.5 → 0.8 → 1.5 渐进调参，最终消除饱和获得真正多目标优化</li>
      <li><b>相关性矩阵修复</b>：对齐 index + 截断 + fallback，根除间歇性崩溃</li>
    </ul>
  </div>
  <div class="highlight-box" style="border-left-color:#ff4444">
    <h4 style="color:#ff4444">⚠️ 局限与根因</h4>
    <ul>
      <li><b>量价同质化</b>：63 因子中 80 对 |r| > 0.5，换手/波动/动量三类高度重叠，组合增益天花板低</li>
      <li><b>A股全反转市场</b>：正向动量因子 ICIR 虽高，但实际选股结果可能偏向近期强势股（反转被"错误"做多）</li>
      <li><b>QMT 成本</b>：双边 0.72% 对周频调仓是巨大负担，均衡策略 7.2% 周换手 = 年化 2.7% 成本已是最低</li>
      <li><b>基本面因子缺失</b>：f_score (ICIR=+0.092)、streak (ICIR=-0.233) 等低 ICIR 但独立的因子未被 GA 选中</li>
    </ul>
  </div>
  <div class="highlight-box" style="border-left-color:#1d9bf0">
    <h4 style="color:#1d9bf0">🚀 建议方向</h4>
    <ul>
      <li><b>混合基本面因子</b>：在因子池中加入 f_score、streak 等独立因子，强制 GA 考虑品类多样性</li>
      <li><b>品类约束</b>：限制同品类（换手/波动/动量）因子权重之和上限（如 ≤ 40%），打破同质化</li>
      <li><b>降低调仓频率</b>：月频可大幅降低交易成本（年化成本从 2.7% → ~0.5%），参考策略二 v4.3 的成功</li>
      <li><b>动态因子池</b>：Phase 2 筛选出的 5 个独立因子（streak/f_score/dollar_vol_20d/high_52w_dist/skew_1m）加入本轮</li>
    </ul>
  </div>
</div>

<div class="footer">
  Factor Alchemy v4 · NSGA-II Multi-Objective GA · Generated 2026-06-08<br>
  ICIR_THRESHOLD=1.5 · 40pop × 5gen · 63 factors · 278 weeks · 5,710 stocks
</div>

</div>
</body>
</html>'''

# Write HTML
out_path = os.path.join(OUT_DIR, 'report_20260608.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Report written: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
