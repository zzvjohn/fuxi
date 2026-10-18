from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for old,new in [('fuxi_notice_tests_v4_20261003.py','fuxi_notice_tests_v5_20261003.py'),
                ('fuxi_notice_ten_rounds_v3_20261003.py','fuxi_notice_ten_rounds_v4_20261003.py'),
                ('fuxi_notice_campaign_v3_driver_20261003.py','fuxi_notice_campaign_v4_driver_20261003.py')]:
    s=(ROOT/'scripts'/old).read_bytes().decode('utf-8')
    if 'tests_' in old:s=s.replace('tests_v4','tests_v5')
    elif 'ten_rounds' in old:
        s=s.replace('ten_rounds_v3','ten_rounds_v4').replace('tests_v4','tests_v5')
        s=s.replace('postflight_halt_688766/BUNDLE.json','postflight_halt_interval_688766/BUNDLE.json')
    else:s=s.replace('ten_rounds_v3','ten_rounds_v4').replace('rc10','rc11')
    with (ROOT/'scripts'/new).open('xb') as h:h.write(s.encode('utf-8'))
print('rc11 test and ten-round variants written; rc10 stopped files retained')
