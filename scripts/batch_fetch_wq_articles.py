# -*- coding: utf-8 -*-
"""WQ Brain A股文章批量抓取 → 提取因子逻辑"""
import json, re, sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 文章ID列表（从HTML提取的A股非🔒文章）
ARTICLE_IDS = [
    '19137620283415','18600255773719','19165054672663',
    '19149488926103','19144966455959','19139586296727',
    '19106989395863','19139800986903','19138922802583',
    '19137333421719','19135894021527','19393270812311',
    '19383530804503','19509179177879','19510326634519',
    '20073476508055','20088908680087','20089517507735',
    '20090584979479','20108973243671','20262909489431',
    '21109320413335','21120825510551','21145402272663',
    '22071441537303','22130841774359','22162473785623',
    '22817801744919','19117391714839','19114962669335',
    '20238047751959','20247158390295','20255074372375',
    '24543603570199','26702144521751',
]

# 已抓取的跳过
SKIP = {'18600255773719', '19165054672663', '20088908680087'}

import urllib.request, urllib.error

MCP_URL = 'http://127.0.0.1:8002/mcp'

def mcp_call(method, params):
    """调用WQ Brain MCP"""
    # 初始化
    init_body = json.dumps({
        "jsonrpc":"2.0","id":1,"method":"initialize",
        "params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"batch","version":"1.0"}}
    }).encode()
    req = urllib.request.Request(MCP_URL, data=init_body, headers={
        'Content-Type':'application/json','Accept':'application/json, text/event-stream'
    })
    resp = urllib.request.urlopen(req, timeout=30)
    # 读取session ID (简化: 直接调)
    resp.read()
    
    # 第二次请求
    body = json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/call","params":params}).encode()
    req2 = urllib.request.Request(MCP_URL, data=body, headers={
        'Content-Type':'application/json','Accept':'application/json, text/event-stream'
    })
    resp2 = urllib.request.urlopen(req2, timeout=60)
    raw = resp2.read().decode('utf-8','ignore')
    return raw

results_summary = []
for i, aid in enumerate(ARTICLE_IDS):
    if aid in SKIP: 
        continue
    
    print(f"[{i+1}/{len(ARTICLE_IDS)}] {aid}...", end=' ')
    sys.stdout.flush()
    
    try:
        raw = mcp_call('read_forum_post', {"name":"read_forum_post","arguments":{"article_id":aid}})
        
        data_match = re.search(r'data: ({.*})', raw, re.DOTALL)
        if not data_match:
            print("NO DATA")
            continue
        
        data = json.loads(data_match.group(1))
        result_str = data['result']['content'][0]['text']
        result = json.loads(result_str)
        
        if result.get('error'):
            print(f"ERR: {result['error'][:60]}")
            continue
        
        post = result['post']
        title = post['title']
        body = post['body']
        
        # 保存
        safe_name = title[:30].replace('/','_').replace('\\','_').replace(':','_')
        fname = f"e:/quant/reports/wq_{aid}.txt"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(f"=== {title} ===\n\n")
            f.write(body)
        
        # 提取因子信息
        keywords = {
            '换手率':'turnover','反转':'reversal','动量':'momentum',
            '波动':'volatility','估值':'valuation','价量':'vol_price',
            '振幅':'amplitude','隔夜':'overnight','日内':'intraday',
            '均线':'ma','放量':'volume_break','拥挤':'crowding',
            '跳跃':'jump','信息分布':'info_dist','凸显':'salience',
            'RPV':'rpv','UTR':'utr','STR':'str','UMR':'umr',
            'GTR':'gtr','SCC':'scc','ICIR':'icir',
        }
        found_kw = [kw for kw in keywords if kw in body]
        
        summary = {
            'id': aid, 'title': title, 'len': len(body),
            'keywords': found_kw[:8],
            'has_code': 'expression' in body.lower() or 'alpha =' in body.lower()
        }
        results_summary.append(summary)
        
        print(f"OK {len(body)}ch {found_kw[:4]}")
        
    except Exception as e:
        print(f"FAIL: {str(e)[:60]}")
    
    time.sleep(1)

# 保存汇总
with open('e:/quant/reports/wq_articles_summary.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print(f"\n完成: {len(results_summary)}篇")
for s in results_summary:
    print(f"  {s['id']} | {s['title'][:40]} | {','.join(s['keywords'][:5])}")
