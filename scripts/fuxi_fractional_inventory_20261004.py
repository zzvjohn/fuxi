"""Read the immutable market once and publish small source-review inventory."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
INPUT=ROOT/'reports/fuxi_v1_announcements_20261003/ten_rounds_v4/market_input.json'


def main():
    require(file_hash(INPUT)==load_json(INPUT.parent/'INPUT_RECEIPT.json')['sha256'],'prior market changed')
    market=load_json(INPUT)['market']
    terms=market.get('action_terms',[])
    current=[t for t in terms if '2025-01-02'<=t['ex_date']<='2026-09-30']
    with (OUT/'ACTION_TERM_INVENTORY.json').open('x',encoding='utf-8') as f:json.dump(current,f,ensure_ascii=False,indent=2)
    print(json.dumps({'market_bars':len(market['bars']),'terms':len(current),
        'focus':[t for t in current if t['instrument'] in {'001211.XSHE','688215.XSHG','688557.XSHG','920699.BJ'}]},ensure_ascii=False)[:11000],flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
