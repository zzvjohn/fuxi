"""Point final reporting at the rc11 complete attempt and preserve rc10 STOP."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'scripts/fuxi_notice_finalize_20261003.py';s=p.read_bytes().decode('utf-8')
s=s.replace('tests_v4','tests_v5').replace('ten_rounds_v3','ten_rounds_v4')
s=s.replace('postflight_halt_688766/BUNDLE.json','postflight_halt_interval_688766/BUNDLE.json')
s=s.replace("['2025-11-26']))==1", "['2025-11-26','2025-11-27','2025-11-28','2025-12-01','2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-08']))==9")
s=s.replace('V1.0 rc10','V1.0 rc11').replace('伏羲rc10已购买','伏羲rc11已购买').replace('1.0.0-rc10','1.0.0-rc11')
old='只确认2025-11-26这一天，不从“预计不超过10日”推导后续停牌。'
new='rc10旧尝试仅确认2025-11-26，12预检中反转5账户缺口因此移至2025-11-27；已记录ten_rounds_v3/STOP_FOR_INTERVAL_CORRECTION并停止，完成候选轮数0，未与两组完整十轮混计。补归档2025-12-09复牌原文后，rc11只使用两份原文共同覆盖的2025-11-26..2025-12-08九个实际交易日，不从预计时长推导。起始日原文为11月26日，复牌原文追述为11月25日，一日冲突保留，争议的11月25日不由该包补证。停止日12月8日与复牌日12月9日来自明确表格/条款；后续公告用于研究估值区间的回顾性确认，不声明各历史日独立PIT认证，不作为alpha输入。'
assert old in s;s=s.replace(old,new)
s=s.replace('连续停牌原文接入；','连续停牌原文交集接入并保留日期冲突；')
compile(s,str(p),'exec');p.write_bytes(s.encode('utf-8'))
p=ROOT/'scripts/fuxi_notice_verify_final_20261003.py';s=p.read_bytes().decode('utf-8')
s=s.replace('tests_v4','tests_v5').replace('ten_rounds_v3','ten_rounds_v4')
compile(s,str(p),'exec');p.write_bytes(s.encode('utf-8'))
print('rc11 reporting updated; rc9 ten results and rc10 stopped preflight preserved')
