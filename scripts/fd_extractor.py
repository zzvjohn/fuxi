# -*- coding: utf-8 -*-
"""从 factors.directory 页面 HTML 提取因子数据 (v3 提取器: 双路径).

路径 A: <script type="application/ld+json"> Article (tech 类页面, 含 formulas)
路径 B: Next.js RSC flight payload 的 "factor":{...} 块 (basic-surface 等页面)
  flight 字符串的转义规则: \\" -> ", \\\\ -> \\, \\n 为真实换行
  处理: 反转义前两者后, 括号配平提取 factor 块, json.loads 该块
"""
import json
import re

LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)


def _unescape_js(s):
    return s.replace('\\"', '"').replace('\\\\', '\\')


def _extract_balanced(p, start_brace_idx):
    depth = 0
    i = start_brace_idx
    while i < len(p):
        c = p[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(p[start_brace_idx:i + 1])
                except Exception:
                    return None
        i += 1
    return None


def extract_factor_data(html):
    # 路径 A: LD+JSON
    for m in LD_RE.finditer(html):
        raw = m.group(1)
        for cand in (raw, _unescape_js(raw)):
            try:
                obj = json.loads(cand)
            except Exception:
                continue
            if obj.get("@type") == "Article" and obj.get("formulas"):
                return {
                    "name": obj.get("headline", ""),
                    "tags": obj.get("tags", []),
                    "formulas": [{"text": f.get("text", ""),
                                  "latex": f.get("latex", "")}
                                 for f in obj["formulas"]],
                    "description": obj.get("description", ""),
                }
    # 路径 B: RSC flight payload (结尾模式 '"]\n</script>' 或 '"])</script>')
    for m in re.finditer(r'self\.__next_f\.push\(\[1,"', html):
        start = m.end()
        end = html.find('"])', start)
        if end == -1:
            continue
        decoded = _unescape_js(html[start:end])
        for fm in re.finditer(r'"factor"\s*:\s*\{', decoded):
            fac = _extract_balanced(decoded, fm.end() - 1)
            if fac and fac.get("formulas"):
                return {
                    "name": fac.get("name") or fac.get("title", ""),
                    "tags": fac.get("tags", []),
                    "formulas": [{"text": f.get("text", ""),
                                  "latex": f.get("latex", "")}
                                 for f in fac["formulas"]],
                    "description": fac.get("description", ""),
                }
    return None


if __name__ == "__main__":
    for f in ["data/_fd_tmp/sample.html", "data/_fd_tmp/sample2.html"]:
        html = open(f, encoding="utf-8", errors="replace").read()
        fac = extract_factor_data(html)
        print("=====", f)
        if fac is None:
            print("  FAIL")
            continue
        print("  name:", fac["name"])
        print("  tags:", fac["tags"])
        for fm in fac["formulas"]:
            print("  formula:", fm["text"], "| latex:", fm["latex"][:80])
